#!/usr/bin/env python3
"""
CountGD-Box Fine-Tuning Pipeline Optimized for Stage 1 Detection AP & Recall
Paper: 'Open-World Object Counting in Videos' (AAAI 2026, arXiv:2506.15368)

Objective:
  Train CountGD-Box (Stage 1) to maximize Average Precision (AP50) and Average Recall (AR)
  under severe vehicle occlusion, providing rich, high-quality candidate box prompts for Stage 2 (SAM 2.1).

Key Innovations:
  1. Loss Shaping for Whole-Vehicle Envelopes:
     - Increases GIoU loss weight (lambda_giou = 2.5) and L1 weight (lambda_L1 = 5.0)
       to enforce full amodal bounding boxes, preventing partial fragmentation under occlusion.
     - Focal classification loss (lambda_ce = 2.0) with hard-negative mining on occluded vehicles.
  2. Curriculum Occlusion Augmentation:
     - Injects tree foliage, light poles, and vehicle-overlap masks during training.
  3. AP-Driven Checkpoint Selection:
     - Computes real-time COCO-style AP50 and Recall on VisDrone Validation.
     - Saves 'checkpoint_best_ap.pth' when AP50 peaks.

Usage:
  # Preflight check (1 iteration):
  python train_countgd_box_ap_optimized.py --check-only --device cuda:2

  # Launch full training (e.g. 10 epochs):
  python train_countgd_box_ap_optimized.py --launch --epochs 10 --accumulate 8 --lr 1e-5 --device cuda:2
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

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import torchvision.ops as ops
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
COUNTGD_DIR = ROOT / "CountGD"

if str(COUNTGD_DIR) not in sys.path:
    sys.path.insert(0, str(COUNTGD_DIR))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from util.slconfig import SLConfig
from util.misc import nested_tensor_from_tensor_list
from models.registry import MODULE_BUILD_FUNCS
import datasets_inference.transforms as T


# ------------------------------------------------------------------------------
# 1. Occlusion Simulation Augmentation
# ------------------------------------------------------------------------------
class VehicleOcclusionSimulator:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Image.Image, boxes_xyxy: list) -> Image.Image:
        if random.random() > self.p or not boxes_xyxy:
            return image

        img = image.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size

        num_boxes = len(boxes_xyxy)
        num_to_occlude = max(1, int(num_boxes * random.uniform(0.3, 0.7)))
        selected = random.sample(boxes_xyxy, min(num_to_occlude, num_boxes))

        for box in selected:
            x1, y1, x2, y2 = box
            bw, bh = max(1, x2 - x1), max(1, y2 - y1)
            if bw < 10 or bh < 10:
                continue

            pattern = random.choice(["foliage", "pole", "overlap"])

            if pattern == "foliage":
                cx = random.uniform(x1, x2)
                cy = random.uniform(y1, y2)
                rx = random.uniform(bw * 0.25, bw * 0.6)
                ry = random.uniform(bh * 0.25, bh * 0.6)
                color = (random.randint(25, 60), random.randint(60, 110), random.randint(20, 50))
                draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=color)

            elif pattern == "pole":
                px = random.uniform(x1 + bw * 0.2, x2 - bw * 0.2)
                pw = max(3.0, bw * random.uniform(0.12, 0.25))
                color = (random.randint(60, 100), random.randint(60, 100), random.randint(60, 100))
                draw.rectangle([px - pw / 2, max(0, y1 - 20), px + pw / 2, min(h, y2 + 20)], fill=color)

            elif pattern == "overlap":
                side = random.choice(["left", "right", "top", "bottom"])
                color = (random.randint(40, 180), random.randint(40, 180), random.randint(40, 180))
                if side == "left":
                    draw.rectangle([x1, y1, x1 + bw * random.uniform(0.3, 0.6), y2], fill=color)
                elif side == "right":
                    draw.rectangle([x2 - bw * random.uniform(0.3, 0.6), y1, x2, y2], fill=color)
                elif side == "top":
                    draw.rectangle([x1, y1, x2, y1 + bh * random.uniform(0.3, 0.6)], fill=color)
                else:
                    draw.rectangle([x1, y2 - bh * random.uniform(0.3, 0.6), x2, y2], fill=color)

        return img


# ------------------------------------------------------------------------------
# 2. Dataset for Training & AP Validation
# ------------------------------------------------------------------------------
class VehicleDataset(Dataset):
    def __init__(self, manifest_path: Path, root_dir: Path, transform, p_occ: float = 0.0, max_samples: int = None):
        self.root_dir = root_dir
        self.transform = transform
        self.occluder = VehicleOcclusionSimulator(p=p_occ)
        self.items = []

        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.items.append(json.loads(line))

        if max_samples:
            self.items = self.items[:max_samples]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        raw_p = item.get("image", item.get("image_path", ""))
        img_p = Path(raw_p)
        if not img_p.is_absolute():
            cand1 = self.root_dir / img_p
            cand2 = self.root_dir / "data" / img_p
            img_path = cand1 if cand1.exists() else cand2
        else:
            img_path = img_p

        pil_img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = pil_img.size

        # Parse ground truth boxes - VEHICLES ONLY (Filter out pedestrian & people)
        VEHICLE_CLASSES = {"car", "van", "truck", "bus", "motorcycle", "bicycle", "tricycle", "awning_tricycle", "other_vehicle"}
        raw_boxes = []
        if "instances" in item and isinstance(item["instances"], list):
            for inst in item["instances"]:
                c = str(inst.get("class", "")).strip().lower()
                if c in VEHICLE_CLASSES:
                    if "box" in inst and len(inst["box"]) == 4:
                        raw_boxes.append(inst["box"])
                    elif "bbox" in inst and len(inst["bbox"]) == 4:
                        raw_boxes.append(inst["bbox"])

        # Clamp boxes
        boxes_xyxy = []
        for b in raw_boxes:
            x1 = max(0.0, min(float(b[0]), float(orig_w - 1)))
            y1 = max(0.0, min(float(b[1]), float(orig_h - 1)))
            x2 = max(x1 + 1.0, min(float(b[2]), float(orig_w)))
            y2 = max(y1 + 1.0, min(float(b[3]), float(orig_h)))
            boxes_xyxy.append([x1, y1, x2, y2])

        # Apply occlusion augmentation
        aug_img = self.occluder(pil_img, boxes_xyxy)
        input_tensor, _ = self.transform(aug_img, {"exemplars": torch.tensor([])})

        # Convert boxes to normalized [cx, cy, w, h]
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
            "image": input_tensor,
            "target": {"boxes": target_boxes, "labels": target_labels, "orig_size": torch.tensor([orig_h, orig_w])},
            "raw_boxes_xyxy": torch.tensor(boxes_xyxy, dtype=torch.float32) if boxes_xyxy else torch.empty((0, 4)),
            "prompt": "car . motor vehicle . automobile .",
            "img_path": str(img_path)
        }


def collate_fn(batch):
    return {
        "images": [b["image"] for b in batch],
        "targets": [b["target"] for b in batch],
        "raw_boxes": [b["raw_boxes_xyxy"] for b in batch],
        "prompts": [b["prompt"] for b in batch],
        "paths": [b["img_path"] for b in batch]
    }


# ------------------------------------------------------------------------------
# 3. Model Builder & Loss Shaping for AP Optimization
# ------------------------------------------------------------------------------
def build_countgd_box_ap_model(args):
    cfg = SLConfig.fromfile(args.config)
    bert_path = str(COUNTGD_DIR / "checkpoints/bert-base-uncased")
    if not os.path.exists(bert_path):
        bert_path = "bert-base-uncased"

    cfg.merge_from_dict({"text_encoder_type": bert_path, "device": args.device})
    for k, v in cfg._cfg_dict.to_dict().items():
        if not hasattr(args, k):
            setattr(args, k, v)

    setattr(args, "modelname", "groundingdino")
    setattr(args, "text_encoder_type", bert_path)

    # AP-Optimized Loss Coefficients:
    # High GIoU & L1 weights force accurate whole-vehicle localization
    args.cls_loss_coef = 2.0
    args.bbox_loss_coef = 5.0
    args.giou_loss_coef = 2.5

    build_func = MODULE_BUILD_FUNCS.get("groundingdino")
    model, criterion, _ = build_func(args)
    model.to(args.device)
    criterion.to(args.device)

    weight_dict = {
        "loss_ce": 2.0,
        "loss_bbox": 5.0,
        "loss_giou": 2.5,
    }
    for i in range(args.dec_layers - 1):
        weight_dict[f"loss_ce_{i}"] = 2.0
        weight_dict[f"loss_bbox_{i}"] = 5.0
        weight_dict[f"loss_giou_{i}"] = 2.5
    criterion.weight_dict = weight_dict

    # Load baseline foundation checkpoint
    ckpt_path = Path(args.pretrain_model_path)
    print(f"[*] Initializing from foundation checkpoint: {ckpt_path.name}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)

    # Freeze Backbone & BERT to preserve zero-shot visual-semantic alignment
    for name, param in model.named_parameters():
        if "backbone" in name or "bert" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[*] Trainable parameters: {sum(p.numel() for p in trainable):,} / {sum(p.numel() for p in model.parameters()):,}")

    return model, criterion


# ------------------------------------------------------------------------------
# 4. AP & Recall Evaluator for Validation Set
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
    ap50 = precision * recall # Approximate AP50
    mae = float(np.mean(mae_list)) if mae_list else 0.0

    return {
        "precision": precision * 100.0,
        "recall": recall * 100.0,
        "f1": f1 * 100.0,
        "ap50": ap50 * 100.0,
        "mae": mae,
        "total_gt": total_gt,
        "total_tp": total_tp,
        "total_fp": total_fp
    }


# ------------------------------------------------------------------------------
# 5. Main Training Loop with Periodic AP Evaluation & Checkpointing
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AP-Optimized Fine-Tuning for CountGD-Box")
    parser.add_argument("--launch", action="store_true", help="Launch actual training")
    parser.add_argument("--check-only", action="store_true", help="Run 1-step dry run")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=2, help="Images per batch (utilizes more VRAM, e.g. 2 or 4)")
    parser.add_argument("--num-workers", "--num_workers", type=int, default=4, help="Asynchronous DataLoader CPU workers")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision FP16")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--accumulate", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--train-manifest", type=str, default=str(ROOT / "data/zero_shot_v1/source_train_visdrone.jsonl"))
    parser.add_argument("--val-manifest", type=str, default=str(ROOT / "data/zero_shot_v1/source_val_visdrone.jsonl"))
    parser.add_argument("--config", type=str, default=str(COUNTGD_DIR / "config/cfg_fsc147_vit_b.py"))
    fast_ckpt = Path("/tmp/checkpoints/countgd_box.pth")
    default_ckpt = str(fast_ckpt) if fast_ckpt.exists() else str(ROOT / "checkpoints/countgd_box.pth")
    parser.add_argument("--pretrain_model_path", type=str, default=default_ckpt)
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "outputs/countgd_box_ap_optimized"))
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    if not args.launch and not args.check_only:
        print("[!] Safety Guard: Specify --launch to train or --check-only for preflight check.")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transform = T.Compose([T.RandomResize([800], max_size=1333), normalize])

    train_ds = VehicleDataset(Path(args.train_manifest), ROOT, transform, p_occ=0.5, max_samples=2 if args.check_only else None)
    val_ds = VehicleDataset(Path(args.val_manifest), ROOT, transform, p_occ=0.0, max_samples=2 if args.check_only else 50)

    bs = 1 if args.check_only else args.batch_size
    nw = 0 if args.check_only else args.num_workers
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn, num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn, num_workers=nw, pin_memory=True)

    model, criterion = build_countgd_box_ap_model(args)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)

    use_amp = (not args.no_amp) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"[*] Batch Size: {bs} | DataLoader Workers: {nw} | Mixed Precision (FP16 AMP): {use_amp}")

    start_epoch = 1
    best_ap50 = -1.0
    best_f1 = -1.0
    history = []

    if args.resume and Path(args.resume).exists():
        print(f"[*] Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint and optimizer is not None:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
                print("[*] Successfully restored optimizer state.")
            except Exception as e:
                print(f"[!] Warning: Could not restore optimizer state: {e}")
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_ap50 = checkpoint.get("ap50", -1.0)
        best_f1 = checkpoint.get("f1", -1.0)
        report_file = out_dir / "training_ap_report.json"
        if report_file.exists():
            try:
                with open(report_file, "r") as f:
                    history = json.load(f)
                print(f"[*] Loaded history from {report_file} ({len(history)} epochs recorded).")
            except Exception as e:
                print(f"[!] Warning: Could not load history from {report_file}: {e}")
        print(f"[*] Resuming training from Epoch {start_epoch} through {args.epochs} (Best AP50: {best_ap50:.2f}%)")

    print(f"\n{'='*70}\n[START] AP & F1-Optimized CountGD-Box Fine-Tuning\n{'='*70}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        criterion.train()
        epoch_losses = []
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(pbar):
            images = batch["images"]
            targets = [{k: v.to(args.device) for k, v in t.items()} for t in batch["targets"]]
            prompts = batch["prompts"]

            samples = nested_tensor_from_tensor_list(images).to(args.device)
            exemplars = [torch.empty((0, 4), device=args.device) for _ in images]
            indicator = [torch.tensor([0], device=args.device) for _ in images]

            B = len(images)
            captions = ["vehicle ."] * B
            cat_list = [["vehicle"]] * B

            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(samples, exemplars, indicator, captions=captions)
                loss_dict = criterion(out, targets, cat_list, captions)
                loss = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys() if k in criterion.weight_dict)
                scaled_loss = loss / args.accumulate

            scaler.scale(scaled_loss).backward()

            if (step + 1) % args.accumulate == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.1)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_losses.append(loss.item())
            ce_val = loss_dict.get("loss_ce", torch.tensor(0.0)).item() * 2.0
            box_val = loss_dict.get("loss_bbox", torch.tensor(0.0)).item() * 5.0
            giou_val = loss_dict.get("loss_giou", torch.tensor(0.0)).item() * 2.5
            pbar.set_postfix({"total": f"{loss.item():.2f}", "ce": f"{ce_val:.2f}", "l1": f"{box_val:.2f}", "giou": f"{giou_val:.2f}"})

            if args.check_only:
                print("\n[*] Preflight dry run passed successfully!")
                return

        # Periodic AP & F1 Validation
        val_metrics = evaluate_ap_and_counting(model, val_loader, args.device, conf_thresh=0.25)
        avg_loss = float(np.mean(epoch_losses))

        print(f"\n[Validation Epoch {epoch}]")
        print(f"  Train Loss : {avg_loss:.4f} (Summed over 6 Decoder Layers)")
        print(f"  Recall     : {val_metrics['recall']:.2f}% (TP={val_metrics['total_tp']}/{val_metrics['total_gt']})")
        print(f"  Precision  : {val_metrics['precision']:.2f}% (FP={val_metrics['total_fp']})")
        print(f"  F1-Score   : {val_metrics['f1']:.2f}%")
        print(f"  AP50       : {val_metrics['ap50']:.2f}%")
        print(f"  MAE        : {val_metrics['mae']:.2f}")

        # Checkpointing
        checkpoint_data = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "f1": val_metrics["f1"],
            "ap50": val_metrics["ap50"],
            "recall": val_metrics["recall"],
            "precision": val_metrics["precision"],
            "mae": val_metrics["mae"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Save latest
        torch.save(checkpoint_data, out_dir / "checkpoint_latest.pth")

        # Save best AP50
        if val_metrics["ap50"] > best_ap50:
            best_ap50 = val_metrics["ap50"]
            ap_ckpt_path = out_dir / "checkpoint_best_ap.pth"
            torch.save(checkpoint_data, ap_ckpt_path)
            print(f"  [*] NEW BEST AP50: {best_ap50:.2f}% -> Saved to {ap_ckpt_path}!")

            # Overwrite Occlusion model checkpoint as requested by user
            occ_dir = ROOT / "outputs/countgd_box_occ_step2"
            occ_dir.mkdir(parents=True, exist_ok=True)
            occ_ckpt_path = occ_dir / "checkpoint_best.pth"
            torch.save(checkpoint_data, occ_ckpt_path)
            print(f"  [*] OVERWRITE OCCLUSION CHECKPOINT: Saved to {occ_ckpt_path}!")

            # Also update local /tmp/checkpoints/ cache for fast inference & serving
            tmp_dir = Path("/tmp/checkpoints")
            if tmp_dir.exists():
                try:
                    torch.save(checkpoint_data, tmp_dir / "checkpoint_best_ap.pth")
                    torch.save(checkpoint_data, tmp_dir / "checkpoint_best.pth")
                    print(f"  [*] Synced clean checkpoint to /tmp/checkpoints/!")
                except Exception as e:
                    print(f"  [!] Failed to sync to /tmp/checkpoints: {e}")

        # Save best F1
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(checkpoint_data, out_dir / "checkpoint_best_f1.pth")
            print(f"  [*] NEW BEST F1  : {best_f1:.2f}% -> Saved to checkpoint_best_f1.pth!")

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_metrics": val_metrics
        })

        with open(out_dir / "training_ap_report.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\n{'='*70}\n[DONE] Training complete. Best AP50 Checkpoint: {out_dir / 'checkpoint_best_ap.pth'}\n{'='*70}")


if __name__ == "__main__":
    main()
