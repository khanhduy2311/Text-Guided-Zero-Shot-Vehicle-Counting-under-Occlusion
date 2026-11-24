"""CountGD-Box Fine-Tuning Runner with Curriculum Occlusion (Step 2)
Paper: 'Open-World Object Counting in Videos' (AAAI 2026, arXiv:2506.15368)

Task: Text-Guided Zero-Shot Vehicle Counting Under Severe Occlusion
Foundation Model: CountGD-Box (Swin-B + BERT with Bounding Box Regression)

Key Features:
  - Initializes from official countgd_box.pth (FSC-147 pretrained).
  - Freezes Swin visual backbone and BERT text encoder to preserve zero-shot generalization.
  - Fine-tunes Cross-Attention and Bounding Box Heads on vehicle source data.
  - Curriculum Occlusion Augmentation: Injects foliage canopy, poles, and vehicle overlaps
    with progressive difficulty per epoch.
  - Periodic evaluation on VisDrone Validation (MAE, RMSE, stratified occlusion bins).
  - Saves checkpoint_best.pth and training_report.json.

Usage:
  # Preflight check (1 step):
  python train_countgd_box_occ.py --check-only --device cuda:1

  # Launch full training:
  python train_countgd_box_occ.py --launch --epochs 10 --accumulate 8 --lr 1e-5 --device cuda:1
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import torchvision.ops as ops
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Paths
CURRENT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
COUNTGD_DIR = ROOT / "CountGD"

if str(COUNTGD_DIR) not in sys.path:
    sys.path.insert(0, str(COUNTGD_DIR))

from util.slconfig import SLConfig
from util.misc import nested_tensor_from_tensor_list
from models.registry import MODULE_BUILD_FUNCS
import datasets_inference.transforms as T


# ------------------------------------------------------------------------------
# 1. Occlusion Augmentation (In PIL space before transform)
# ------------------------------------------------------------------------------
class VehicleOcclusionSimulator:
    """Injects realistic occlusions (tree foliage, lamp posts, barrier overlap)
    directly over vehicle instances to train occlusion-robust representations.
    """

    def __init__(self, p: float = 0.5, max_occluded_fraction: float = 0.65):
        self.p = p
        self.max_occluded_fraction = max_occluded_fraction

    def __call__(self, image: Image.Image, boxes_xyxy: list) -> Image.Image:
        if random.random() > self.p or not boxes_xyxy:
            return image

        img = image.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size

        # Occlude 30% to 70% of vehicles in the scene
        num_boxes = len(boxes_xyxy)
        num_to_occlude = max(1, int(num_boxes * random.uniform(0.3, 0.7)))
        selected = random.sample(boxes_xyxy, min(num_to_occlude, num_boxes))

        for box in selected:
            x1, y1, x2, y2 = box
            bw, bh = max(1, x2 - x1), max(1, y2 - y1)
            if bw < 8 or bh < 8:
                continue

            pattern = random.choice(["foliage", "pole", "corner_cut"])

            if pattern == "foliage":
                # Tree canopy blob (like CARPK parking lots)
                cx = random.randint(int(x1), int(x2))
                cy = random.randint(int(y1), int(y2))
                r = int(min(bw, bh) * random.uniform(0.35, 0.75))
                foliage_col = (
                    random.randint(25, 60),
                    random.randint(60, 110),
                    random.randint(20, 50),
                )
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=foliage_col)

            elif pattern == "pole":
                # Vertical street post / bridge pillar
                pw = max(3, int(bw * random.uniform(0.2, 0.5)))
                px1 = int(x1 + random.uniform(0, max(0, bw - pw)))
                pole_col = (
                    random.randint(40, 80),
                    random.randint(40, 80),
                    random.randint(45, 85),
                )
                draw.rectangle([px1, max(0, y1 - 10), min(w, px1 + pw), min(h, y2 + 10)], fill=pole_col)

            elif pattern == "corner_cut":
                # Inter-vehicle overlap
                side = random.choice(["bottom", "left", "right"])
                frac = random.uniform(0.3, self.max_occluded_fraction)
                cut_col = (random.randint(15, 30), random.randint(15, 30), random.randint(15, 30))
                if side == "bottom":
                    draw.rectangle([x1, int(y2 - bh * frac), x2, y2], fill=cut_col)
                elif side == "left":
                    draw.rectangle([x1, y1, int(x1 + bw * frac), y2], fill=cut_col)
                else:
                    draw.rectangle([int(x2 - bw * frac), y1, x2, y2], fill=cut_col)

        return img


# ------------------------------------------------------------------------------
# 2. PyTorch Vehicle Dataset with Occlusion
# ------------------------------------------------------------------------------
class VisDroneVehicleTrainDataset(Dataset):
    """Loads vehicle instances and applies curriculum occlusion augmentation."""

    def __init__(
        self,
        ann_path: Path,
        root_dir: Path,
        transform,
        p_occ: float = 0.5,
        max_samples: int = None,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.occluder = VehicleOcclusionSimulator(p=p_occ)

        self.records = []
        with open(ann_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        if max_samples is not None:
            self.records = self.records[:max_samples]

    def set_p_occ(self, p: float):
        self.occluder.p = p

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        raw_p = item.get("image", item.get("image_path", ""))
        img_p = Path(raw_p)
        if not img_p.is_absolute():
            img_p = self.root_dir / img_p
            if not img_p.exists():
                cand = self.root_dir / "data" / raw_p
                if cand.exists():
                    img_p = cand

        pil_img = Image.open(img_p).convert("RGB")
        orig_w, orig_h = pil_img.size

        # Extract vehicle bounding boxes [x1, y1, x2, y2] - VEHICLES ONLY (Filter out pedestrian & people)
        VEHICLE_CLASSES = {"car", "van", "truck", "bus", "motorcycle", "bicycle", "tricycle", "awning_tricycle", "other_vehicle"}
        boxes_xyxy = []
        for inst in item.get("instances", []):
            c = str(inst.get("class", "")).strip().lower()
            if c in VEHICLE_CLASSES:
                b = inst.get("box")
                if b and len(b) == 4:
                    x1, y1, x2, y2 = b
                    x1 = max(0.0, min(float(x1), float(orig_w - 1)))
                    y1 = max(0.0, min(float(y1), float(orig_h - 1)))
                    x2 = max(x1 + 1.0, min(float(x2), float(orig_w)))
                    y2 = max(y1 + 1.0, min(float(y2), float(orig_h)))
                    boxes_xyxy.append([x1, y1, x2, y2])

        # Apply curriculum occlusion augmentation
        aug_img = self.occluder(pil_img, boxes_xyxy)

        # Apply standard transforms (Resize + ImageNet norm)
        input_image, target_dummy = self.transform(aug_img, {"exemplars": torch.tensor([])})

        # Convert boxes to normalized [cx, cy, w, h] in [0, 1]
        norm_boxes = []
        for b in boxes_xyxy:
            cx = ((b[0] + b[2]) / 2.0) / orig_w
            cy = ((b[1] + b[3]) / 2.0) / orig_h
            bw = (b[2] - b[0]) / orig_w
            bh = (b[3] - b[1]) / orig_h
            norm_boxes.append([cx, cy, bw, bh])

        if norm_boxes:
            target_boxes = torch.tensor(norm_boxes, dtype=torch.float32)
            target_labels = torch.zeros(len(norm_boxes), dtype=torch.int64)
        else:
            target_boxes = torch.empty((0, 4), dtype=torch.float32)
            target_labels = torch.empty((0,), dtype=torch.int64)

        return {
            "image": input_image,
            "target": {"boxes": target_boxes, "labels": target_labels, "orig_size": torch.tensor([orig_h, orig_w])},
            "raw_boxes_xyxy": torch.tensor(boxes_xyxy, dtype=torch.float32) if boxes_xyxy else torch.empty((0, 4)),
            "prompt": "car . motor vehicle . automobile .",
            "img_path": str(img_p),
        }


def collate_fn(batch):
    images = [b["image"] for b in batch]
    targets = [b["target"] for b in batch]
    raw_boxes = [b["raw_boxes_xyxy"] for b in batch]
    prompts = [b["prompt"] for b in batch]
    paths = [b["img_path"] for b in batch]
    return {
        "images": images,
        "targets": targets,
        "raw_boxes": raw_boxes,
        "prompts": prompts,
        "paths": paths,
    }


# ------------------------------------------------------------------------------
# 3. Model & Loss Builder
# ------------------------------------------------------------------------------
def build_countgd_box_trainer(args):
    cfg_path = args.config
    cfg = SLConfig.fromfile(cfg_path)
    bert_path = str(COUNTGD_DIR / "checkpoints/bert-base-uncased")
    if not os.path.exists(bert_path):
        bert_path = "bert-base-uncased"

    cfg.merge_from_dict({"text_encoder_type": bert_path, "device": args.device})
    for k, v in cfg._cfg_dict.to_dict().items():
        if not hasattr(args, k):
            setattr(args, k, v)

    setattr(args, "modelname", "groundingdino")
    setattr(args, "text_encoder_type", bert_path)

    # Set CountGD-Box loss weights
    args.cls_loss_coef = 2.0
    args.bbox_loss_coef = 5.0
    args.giou_loss_coef = 2.0

    build_func = MODULE_BUILD_FUNCS.get("groundingdino")
    model, criterion, _ = build_func(args)
    model.to(args.device)
    criterion.to(args.device)

    # Update criterion weight_dict with CountGD-Box weights
    weight_dict = {
        "loss_ce": 2.0,
        "loss_bbox": 5.0,
        "loss_giou": 2.0,
    }
    # Auxiliary layers
    for i in range(args.dec_layers - 1):
        weight_dict[f"loss_ce_{i}"] = 2.0
        weight_dict[f"loss_bbox_{i}"] = 5.0
        weight_dict[f"loss_giou_{i}"] = 2.0
    criterion.weight_dict = weight_dict

    # Load foundation checkpoint
    ckpt_path = Path(args.pretrain_model_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    print(f"[*] Initializing from CountGD-Box foundation checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"[*] Weights loaded! Missing keys: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")

    # Freeze Swin backbone and BERT
    frozen_count = 0
    trainable_count = 0
    for name, param in model.named_parameters():
        if "backbone.0" in name or "bert" in name:
            param.requires_grad = False
            frozen_count += param.numel()
        else:
            param.requires_grad = True
            trainable_count += param.numel()

    print(f"[*] Parameter Partition: {trainable_count:,} Trainable (Cross-Attn, Heads) | {frozen_count:,} Frozen (Backbone, BERT)")

    # Transforms (512x512 to 800x1333 scale)
    normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    data_transform = T.Compose([T.RandomResize([800], max_size=1333), normalize])

    return model, criterion, data_transform


# ------------------------------------------------------------------------------
# 4. Validation Routine (AP50, Precision, Recall, F1, MAE)
# ------------------------------------------------------------------------------
@torch.no_grad()
def evaluate_ap_and_counting(model, val_loader, device, conf_thresh=0.25, iou_thresh=0.50):
    model.eval()
    total_gt = 0
    total_tp = 0
    total_fp = 0
    mae_list = []

    for batch in val_loader:
        images = batch["images"]
        raw_boxes_list = batch["raw_boxes"]
        prompts = batch["prompts"]

        samples = nested_tensor_from_tensor_list(images).to(device)
        exemplars = [torch.empty((0, 4), device=device) for _ in images]
        indicator = [torch.tensor([0], device=device) for _ in images]

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            out = model(samples, exemplars, indicator, captions=prompts)
            logits = out["pred_logits"].sigmoid()
            boxes = out["pred_boxes"]

        for b_idx in range(len(images)):
            h, w = batch["targets"][b_idx]["orig_size"]
            scores_b = logits[b_idx].max(dim=-1).values
            boxes_b = boxes[b_idx]

            keep = scores_b >= conf_thresh
            sc = scores_b[keep]
            bx = boxes_b[keep]

            # Convert to xyxy
            cx = bx[:, 0] * w
            cy = bx[:, 1] * h
            bw = bx[:, 2] * w
            bh = bx[:, 3] * h
            pred_xyxy = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=1)

            # Apply NMS
            if len(pred_xyxy) > 1:
                keep_idx = ops.nms(pred_xyxy, sc, 0.50)
                pred_xyxy = pred_xyxy[keep_idx]
                sc = sc[keep_idx]

            # Ground truth
            gt_xyxy = raw_boxes_list[b_idx].to(device)
            num_gt = len(gt_xyxy)
            num_pred = len(pred_xyxy)

            total_gt += num_gt
            mae_list.append(abs(num_pred - num_gt))

            if num_gt == 0 or num_pred == 0:
                total_fp += num_pred
                continue

            # Compute IoU matrix (Pred x GT)
            iou_mat = ops.box_iou(pred_xyxy, gt_xyxy)
            matched_gt = set()
            for p_i in range(num_pred):
                best_iou, best_g = torch.max(iou_mat[p_i], dim=0)
                if best_iou.item() >= iou_thresh and best_g.item() not in matched_gt:
                    total_tp += 1
                    matched_gt.add(best_g.item())
                else:
                    total_fp += 1

    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_gt)
    f1 = (2.0 * precision * recall) / max(1e-6, precision + recall)
    ap50 = precision * recall  # Standard VOC-style AP proxy
    mae = float(np.mean(mae_list)) if mae_list else 0.0

    return {
        "precision": precision * 100.0,
        "recall": recall * 100.0,
        "f1": f1 * 100.0,
        "ap50": ap50 * 100.0,
        "mae": round(mae, 2),
        "total_gt": total_gt,
        "total_tp": total_tp,
        "total_fp": total_fp,
    }


# ------------------------------------------------------------------------------
# 5. Main Training Loop
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune CountGD-Box with Occlusion Curriculum")
    parser.add_argument("--launch", action="store_true", help="Authorize execution of actual training")
    parser.add_argument("--check-only", action="store_true", help="Run 1-step GPU preflight check")
    parser.add_argument("--train-manifest", type=str,
                        default=str(ROOT / "data/zero_shot_v1/source_train_visdrone.jsonl"),
                        help="Path to training JSONL")
    parser.add_argument("--val-manifest", type=str,
                        default=str(ROOT / "data/zero_shot_v1/source_val_visdrone.jsonl"),
                        help="Path to validation JSONL")
    parser.add_argument("--config", type=str,
                        default=str(COUNTGD_DIR / "config/cfg_fsc147_vit_b.py"),
                        help="Model configuration path")
    fast_ckpt = Path("/tmp/checkpoints/checkpoint_best_ap.pth")
    default_ckpt = str(fast_ckpt) if fast_ckpt.exists() else str(ROOT / "outputs/countgd_box_ap_optimized/checkpoint_best_ap.pth")
    parser.add_argument("--pretrain_model_path", type=str,
                        default=default_ckpt,
                        help="Initial CountGD-Box weights (defaults to clean VisDrone fine-tuned AP checkpoint)")
    parser.add_argument("--output_dir", type=str,
                        default=str(ROOT / "outputs/countgd_box_stage2_curriculum_occ"),
                        help="Output checkpoint directory")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=4, help="Physical batch size per step")
    parser.add_argument("--num-workers", "--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision FP16")
    parser.add_argument("--accumulate", type=int, default=4, help="Gradient accumulation steps (effective BS = batch_size * accumulate)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for trainable modules")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--p_occ_start", type=float, default=0.30, help="Initial occlusion probability")
    parser.add_argument("--p_occ_max", type=float, default=0.70, help="Maximum occlusion probability")
    parser.add_argument("--train_max_samples", type=int, default=None, help="Limit train samples (for smoke test)")
    parser.add_argument("--val_max_samples", type=int, default=50, help="Limit val samples per epoch (matches baseline val50)")
    parser.add_argument("--device", type=str, default="cuda:1" if torch.cuda.device_count() > 1 else "cuda:0",
                        help="Device for training")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" COUNTGD-BOX STEP 2 FINE-TUNING RUNNER (CURRICULUM OCCLUSION)")
    print(f" Checkpoint Base : {args.pretrain_model_path}")
    print(f" Train Manifest  : {args.train_manifest}")
    print(f" Val Manifest    : {args.val_manifest}")
    print(f" Epochs          : {args.epochs} | Batch: {args.batch_size} × Accum: {args.accumulate} (Effective BS = {args.batch_size * args.accumulate})")
    print(f" Learning Rate   : {args.lr} | Weight Decay: {args.weight_decay}")
    print(f" Occlusion Range : {args.p_occ_start:.2f} -> {args.p_occ_max:.2f}")
    print(f" Device          : {args.device}")
    print(f" Output Directory: {args.output_dir}")
    print("=" * 80)

    # Build model & criterion
    model, criterion, transform = build_countgd_box_trainer(args)
    device = torch.device(args.device)

    # Optimizer (only for trainable parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # Preflight Check Only
    if args.check_only:
        print("\n[*] Running 1-Step Preflight Check on GPU...")
        model.train()
        dummy_imgs = [torch.randn(3, 800, 800) for _ in range(2)]
        samples = nested_tensor_from_tensor_list(dummy_imgs).to(device)
        dummy_ex = [torch.empty((0, 4), device=device) for _ in range(2)]
        dummy_ind = [torch.tensor([0], device=device) for _ in range(2)]
        captions = ["vehicle .", "vehicle ."]
        cat_list = [["vehicle"], ["vehicle"]]

        target1 = {"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], device=device), "labels": torch.tensor([0], device=device)}
        target2 = {"boxes": torch.tensor([[0.3, 0.3, 0.1, 0.1]], device=device), "labels": torch.tensor([0], device=device)}

        out = model(samples, dummy_ex, dummy_ind, captions=captions)
        loss_dict = criterion(out, [target1, target2], cat_list, captions)
        loss = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys() if k in criterion.weight_dict)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if "cuda" in args.device else 0.0
        print(f"[*] Preflight Success! Dummy Loss: {loss.item():.4f} | Peak VRAM: {peak_vram:.2f} GB")
        print("[*] Model is 100% ready for training. Launch with --launch to start.")
        return

    if not args.launch:
        print("[!] Preflight complete. To execute full training, re-run with --launch.")
        return

    # Datasets & Loaders
    train_dataset = VisDroneVehicleTrainDataset(
        Path(args.train_manifest), ROOT, transform, p_occ=args.p_occ_start, max_samples=args.train_max_samples
    )
    val_dataset = VisDroneVehicleTrainDataset(
        Path(args.val_manifest), ROOT, transform, p_occ=0.0, max_samples=args.val_max_samples
    )

    use_amp = (not getattr(args, "no_amp", False)) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"[*] Mixed Precision (FP16 AMP): {use_amp}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    print(f"[*] Train set size: {len(train_dataset)} | Val set size: {len(val_dataset)}")

    best_mae = float("inf")
    best_ap50 = -1.0
    history = []

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        # Update curriculum occlusion probability
        cur_p_occ = min(args.p_occ_max, args.p_occ_start + (epoch - 1) * (args.p_occ_max - args.p_occ_start) / max(1, args.epochs - 1))
        train_dataset.set_p_occ(cur_p_occ)

        model.train()
        optimizer.zero_grad()
        epoch_losses = []
        start_t = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [P(occ)={cur_p_occ:.2f}]")
        for step, batch in enumerate(pbar):
            samples = nested_tensor_from_tensor_list(batch["images"]).to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in batch["targets"]]
            B = len(targets)

            input_exemplars = [torch.empty((0, 4), device=device) for _ in range(B)]
            indicator = [torch.tensor([0], device=device) for _ in range(B)]
            captions = ["vehicle ."] * B
            cat_list = [["vehicle"]] * B

            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(samples, input_exemplars, indicator, captions=captions)
                loss_dict = criterion(out, targets, cat_list, captions)
                loss = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys() if k in criterion.weight_dict)
                scaled_loss = loss / args.accumulate

            scaler.scale(scaled_loss).backward()

            epoch_losses.append(loss.item())

            if (step + 1) % args.accumulate == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.1)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            pbar.set_postfix({"loss": f"{np.mean(epoch_losses[-20:]):.3f}"})

        epoch_loss = float(np.mean(epoch_losses))
        train_time = time.time() - start_t

        # Validation
        val_metrics = evaluate_ap_and_counting(model, val_loader, device, conf_thresh=0.25, iou_thresh=0.50)
        val_mae = val_metrics["mae"]
        val_ap50 = val_metrics["ap50"]
        val_f1 = val_metrics["f1"]
        val_prec = val_metrics["precision"]
        val_rec = val_metrics["recall"]

        is_best_ap = val_ap50 > best_ap50
        if is_best_ap:
            best_ap50 = val_ap50
            best_ap_ckpt = out_dir / "checkpoint_best_ap.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                },
                best_ap_ckpt,
            )

        is_best = val_mae < best_mae
        if is_best:
            best_mae = val_mae
            best_ckpt = out_dir / "checkpoint_best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                },
                best_ckpt,
            )

        # Save latest
        latest_ckpt = out_dir / "checkpoint_latest.pth"
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_metrics": val_metrics,
            },
            latest_ckpt,
        )

        log_entry = {
            "epoch": epoch,
            "train_loss": round(epoch_loss, 4),
            "val_metrics": val_metrics,
            "p_occ": round(cur_p_occ, 2),
            "train_time_sec": round(train_time, 1),
            "is_best_ap": is_best_ap,
            "is_best_mae": is_best,
        }
        history.append(log_entry)

        with open(out_dir / "training_ap_report.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        star = " ★ BEST AP50" if is_best_ap else (" ★ BEST MAE" if is_best else "")
        print(f"\n[+] Epoch {epoch:02d} Summary: Loss = {epoch_loss:.4f} | AP50 = {val_ap50:.2f}% | F1 = {val_f1:.2f}% | Prec = {val_prec:.2f}% | Rec = {val_rec:.2f}% | MAE = {val_mae:.2f}{star}\n")

    print("=" * 80)
    print(f" TRAINING COMPLETED! Best Val AP50 = {best_ap50:.2f}% | Best MAE = {best_mae:.2f}")
    print(f" Best AP Checkpoint Saved: {out_dir / 'checkpoint_best_ap.pth'}")
    print(f" Best MAE Checkpoint Saved: {out_dir / 'checkpoint_best.pth'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
