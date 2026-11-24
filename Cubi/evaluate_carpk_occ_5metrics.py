"""CARPK-OCC Benchmark Evaluator for 5 Metrics:
AP50, F1-Score, Precision, Recall, MAE
Compares Baseline CountGD-Box (Official FSC-147) vs. Stage 1 CountGD-Box (Clean Vehicles 5 Epochs)
Across 4 Occlusion Tiers: Clean, Mild, Moderate, Heavy.
"""

import argparse
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
import torchvision.ops as ops

CURRENT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
COUNTGD_DIR = ROOT / "CountGD"

if str(COUNTGD_DIR) not in sys.path:
    sys.path.insert(0, str(COUNTGD_DIR))

from util.slconfig import SLConfig
from models.registry import MODULE_BUILD_FUNCS
import datasets_inference.transforms as T


def apply_synthetic_occlusion(image: Image.Image, boxes: list, level: str, seed: int) -> Image.Image:
    """Applies standardized synthetic occlusion matching CARPK-OCC benchmark protocol."""
    if level == "clean" or not boxes:
        return image.copy()

    rng = random.Random(seed)
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    if level == "mild":
        p_occ = 0.35
        frac_min, frac_max = 0.15, 0.30
    elif level == "moderate":
        p_occ = 0.65
        frac_min, frac_max = 0.30, 0.50
    elif level == "heavy":
        p_occ = 0.85
        frac_min, frac_max = 0.50, 0.75
    else:
        return image.copy()

    num_to_occlude = max(1, int(len(boxes) * p_occ))
    selected_boxes = rng.sample(boxes, min(len(boxes), num_to_occlude))

    for box in selected_boxes:
        x1, y1, x2, y2 = box
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        frac = rng.uniform(frac_min, frac_max)

        occ_style = rng.choice(["foliage", "pillar", "shadow_corner"])
        if occ_style == "foliage":
            cx = x1 + rng.uniform(0.2, 0.8) * bw
            cy = y1 + rng.uniform(0.2, 0.8) * bh
            r = max(bw, bh) * frac * 0.7
            foliage_color = (
                rng.randint(25, 55),
                rng.randint(55, 95),
                rng.randint(20, 45),
            )
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=foliage_color)
        elif occ_style == "pillar":
            pw = max(4, int(bw * frac))
            px1 = x1 + rng.uniform(0, max(0, bw - pw))
            pillar_color = (rng.randint(60, 90), rng.randint(60, 90), rng.randint(60, 90))
            draw.rectangle([px1, max(0, y1 - 15), px1 + pw, min(image.height, y2 + 15)], fill=pillar_color)
        else:
            side = rng.choice(["bottom", "left", "right"])
            fill_c = (rng.randint(15, 30), rng.randint(15, 30), rng.randint(15, 30))
            if side == "bottom":
                draw.rectangle([x1, int(y2 - bh * frac), x2, y2], fill=fill_c)
            elif side == "left":
                draw.rectangle([x1, y1, int(x1 + bw * frac), y2], fill=fill_c)
            else:
                draw.rectangle([int(x2 - bw * frac), y1, x2, y2], fill=fill_c)

    return img_copy


def build_model(config_path, checkpoint_path, device):
    cfg = SLConfig.fromfile(config_path)
    bert_path = str(COUNTGD_DIR / "checkpoints/bert-base-uncased")
    if not os.path.exists(bert_path):
        bert_path = "bert-base-uncased"
    cfg.merge_from_dict({"text_encoder_type": bert_path, "device": device})

    # Prepare dummy args
    class DummyArgs:
        pass
    args = DummyArgs()
    for k, v in cfg._cfg_dict.to_dict().items():
        setattr(args, k, v)
    setattr(args, "modelname", "groundingdino")
    setattr(args, "text_encoder_type", bert_path)
    setattr(args, "device", device)

    build_func = MODULE_BUILD_FUNCS.get("groundingdino")
    model, _, _ = build_func(args)
    model.to(device)

    print(f"[*] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        normalize,
    ])
    return model, transform


@torch.no_grad()
def evaluate_model_on_carpk(model, transform, dataset_entries, device, conf_thresh=0.25, iou_thresh=0.50):
    levels = ["clean", "mild", "moderate", "heavy"]
    results_by_level = {lvl: {"tp": 0, "fp": 0, "gt": 0, "mae_list": []} for lvl in levels}

    caption = "car . vehicle ."

    for idx, entry in enumerate(dataset_entries):
        img_path = entry.get("abs_path")
        if not os.path.exists(img_path):
            img_path = str(ROOT / entry.get("filename", ""))

        if not os.path.exists(img_path):
            continue

        orig_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = orig_pil.size
        gt_boxes_xyxy = [inst["bbox"] for inst in entry.get("detection", {}).get("instances", [])]
        num_gt = len(gt_boxes_xyxy)
        gt_tensor = torch.tensor(gt_boxes_xyxy, dtype=torch.float32, device=device) if gt_boxes_xyxy else torch.empty((0, 4), device=device)

        for lvl in levels:
            # Generate deterministic occlusion per image + level
            seed = 42 + idx * 100 + levels.index(lvl)
            occ_img = apply_synthetic_occlusion(orig_pil, gt_boxes_xyxy, lvl, seed=seed)

            # Transform
            img_tensor, _ = transform(occ_img, {"exemplars": torch.zeros((0, 4))})
            img_tensor = img_tensor.unsqueeze(0).to(device)

            with torch.amp.autocast("cuda", enabled=("cuda" in str(device))):
                out = model(
                    img_tensor,
                    [torch.zeros((0, 4), device=device)],
                    [torch.tensor([0], device=device)],
                    captions=[caption],
                )

            logits = out["pred_logits"][0].sigmoid()
            scores = logits.max(dim=-1).values
            boxes = out["pred_boxes"][0]

            keep = scores >= conf_thresh
            sc = scores[keep]
            bx = boxes[keep]

            # Convert to xyxy
            cx = bx[:, 0] * orig_w
            cy = bx[:, 1] * orig_h
            bw = bx[:, 2] * orig_w
            bh = bx[:, 3] * orig_h
            pred_xyxy = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=1)

            # Apply NMS
            if len(pred_xyxy) > 1:
                keep_idx = ops.nms(pred_xyxy, sc, 0.50)
                pred_xyxy = pred_xyxy[keep_idx]
                sc = sc[keep_idx]

            num_pred = len(pred_xyxy)
            results_by_level[lvl]["gt"] += num_gt
            results_by_level[lvl]["mae_list"].append(abs(num_pred - num_gt))

            if num_gt == 0 or num_pred == 0:
                results_by_level[lvl]["fp"] += num_pred
                continue

            iou_mat = ops.box_iou(pred_xyxy, gt_tensor)
            matched_gt = set()
            for p_i in range(num_pred):
                best_iou, best_g = torch.max(iou_mat[p_i], dim=0)
                if best_iou.item() >= iou_thresh and best_g.item() not in matched_gt:
                    results_by_level[lvl]["tp"] += 1
                    matched_gt.add(best_g.item())
                else:
                    results_by_level[lvl]["fp"] += 1

    # Summarize per level and overall
    summary = {}
    all_tp = 0
    all_fp = 0
    all_gt = 0
    all_mae_list = []

    for lvl in levels:
        tp = results_by_level[lvl]["tp"]
        fp = results_by_level[lvl]["fp"]
        gt = results_by_level[lvl]["gt"]
        mae_arr = results_by_level[lvl]["mae_list"]

        prec = (tp / max(1, tp + fp)) * 100.0
        rec = (tp / max(1, gt)) * 100.0
        f1 = (2.0 * prec * rec) / max(1e-6, prec + rec)
        ap50 = (prec * rec) / 100.0
        mae = float(np.mean(mae_arr)) if mae_arr else 0.0

        summary[lvl] = {
            "ap50": round(ap50, 2),
            "f1": round(f1, 2),
            "precision": round(prec, 2),
            "recall": round(rec, 2),
            "mae": round(mae, 2),
            "tp": tp,
            "fp": fp,
            "gt": gt,
        }

        all_tp += tp
        all_fp += fp
        all_gt += gt
        all_mae_list.extend(mae_arr)

    overall_prec = (all_tp / max(1, all_tp + all_fp)) * 100.0
    overall_rec = (all_tp / max(1, all_gt)) * 100.0
    overall_f1 = (2.0 * overall_prec * overall_rec) / max(1e-6, overall_prec + overall_rec)
    overall_ap50 = (overall_prec * overall_rec) / 100.0
    overall_mae = float(np.mean(all_mae_list)) if all_mae_list else 0.0

    summary["overall"] = {
        "ap50": round(overall_ap50, 2),
        "f1": round(overall_f1, 2),
        "precision": round(overall_prec, 2),
        "recall": round(overall_rec, 2),
        "mae": round(overall_mae, 2),
        "tp": all_tp,
        "fp": all_fp,
        "gt": all_gt,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    cfg_path = str(COUNTGD_DIR / "config/cfg_fsc147_vit_b.py")
    ann_path = str(ROOT / "data/vehicle_val_carpk_holdout.jsonl")

    # Load CARPK holdout samples
    print(f"[*] Reading CARPK holdout dataset: {ann_path}")
    with open(ann_path, "r", encoding="utf-8") as f:
        carpk_entries = [json.loads(line.strip()) for line in f if line.strip()]
    print(f"[*] Total CARPK holdout images: {len(carpk_entries)}")

    models_to_test = [
        {
            "name": "CountGD-Box (Gốc FSC-147)",
            "key": "baseline",
            "ckpt": str(COUNTGD_DIR / "checkpoints/countgd_official.pth"),
        },
        {
            "name": "CountGD-Box (Stage 1 VisDrone Clean)",
            "key": "stage1_clean",
            "ckpt": str(ROOT / "outputs/countgd_box_ap_optimized/checkpoint_best_ap.pth"),
        },
    ]

    all_results = {}

    for m_info in models_to_test:
        print("\n" + "=" * 70)
        print(f" EVALUATING: {m_info['name']}")
        print(f" Checkpoint: {m_info['ckpt']}")
        print("=" * 70)

        t0 = time.time()
        model, transform = build_model(cfg_path, m_info["ckpt"], args.device)
        metrics = evaluate_model_on_carpk(model, transform, carpk_entries, args.device)
        t_cost = time.time() - t0

        print(f"[*] Completed evaluation in {t_cost:.1f}s")
        print(f"[*] Overall AP50: {metrics['overall']['ap50']:.2f}% | F1: {metrics['overall']['f1']:.2f}% | MAE: {metrics['overall']['mae']:.2f}")
        for lvl in ["clean", "mild", "moderate", "heavy"]:
            print(f"    - {lvl.capitalize():<8}: AP50={metrics[lvl]['ap50']:.2f}%, F1={metrics[lvl]['f1']:.2f}%, Prec={metrics[lvl]['precision']:.2f}%, Rec={metrics[lvl]['recall']:.2f}%, MAE={metrics[lvl]['mae']:.2f}")

        all_results[m_info["key"]] = {
            "name": m_info["name"],
            "checkpoint": m_info["ckpt"],
            "metrics": metrics,
            "eval_time_sec": round(t_cost, 1),
        }

        # Clear VRAM between models
        del model
        torch.cuda.empty_cache()

    out_file = ROOT / "outputs/carpk_occ_5metrics_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[+] Saved full CARPK-OCC 5-metric benchmark to: {out_file}")


if __name__ == "__main__":
    main()
