#!/usr/bin/env python3
"""
Comprehensive 5-Dataset Benchmark Suite for CountVImg (CountGD-Box + SAM 2.1)
Evaluates across:
1. CARPK (459 images) -> Normal Counting (MAE, RMSE)
2. CARPK-OCC-LOCAL (459 images x 3 levels: Mild, Moderate, Heavy) -> Occlusion Counting (MAE, RMSE)
3. TRANCOS (421 images) -> Dense Traffic Counting & Localization (MAE, RMSE, Precision, Recall, F1)
4. OVAD (100 images) -> Attribute-Guided Color Counting ("red car") -> Localization F1, Count MAE
5. HICO (100 images) -> Relationship Recognition ("person riding motorcycle") -> Precision, Recall, F1

Author: CS331 Project Pipeline
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.ops as ops
from tqdm import tqdm
from ultralytics import SAM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "CountGD"))
sys.path.insert(0, str(ROOT / "Cubi"))

from util.slconfig import SLConfig
from util.misc import nested_tensor_from_tensor_list
from models.registry import MODULE_BUILD_FUNCS
import datasets_inference.transforms as T


# ------------------------------------------------------------------------------
# 1. Model Loading & Inference Pipeline
# ------------------------------------------------------------------------------
def load_detector_model(ckpt_path: Path, device: str = "cuda:2"):
    cfg_file = ROOT / "CountGD/config/cfg_fsc147_vit_b.py"
    cfg = SLConfig.fromfile(str(cfg_file))
    bert_path = str(ROOT / "CountGD/checkpoints/bert-base-uncased")

    class Args:
        pass
    args = Args()
    for k, v in cfg._cfg_dict.to_dict().items():
        setattr(args, k, v)
    setattr(args, "modelname", "groundingdino")
    setattr(args, "text_encoder_type", bert_path)
    setattr(args, "device", device)

    build_func = MODULE_BUILD_FUNCS.get("groundingdino")
    model, _, _ = build_func(args)
    model.to(device)

    load_path = ckpt_path
    print(f"[*] Loading Detector Checkpoint: {load_path.name} (from {load_path})")
    ckpt = torch.load(str(load_path), map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transform = T.Compose([T.RandomResize([800], max_size=1333), normalize])
    return model, transform


def run_stage1_detector(model, transform, pil_img, device="cuda:2", prompt="car .", conf_thresh=0.20, nms_iou=0.70):
    orig_w, orig_h = pil_img.size
    img_tensor, _ = transform(pil_img, None)
    img_tensor = img_tensor.to(device)
    samples = nested_tensor_from_tensor_list([img_tensor])

    caption = prompt.strip()
    if not caption.endswith("."):
        caption += " ."

    with torch.no_grad():
        out = model(samples, [torch.empty((0, 4), device=device)], [torch.tensor([0], device=device)], captions=[caption])
        logits = out["pred_logits"][0].sigmoid()
        boxes = out["pred_boxes"][0]
        scores = logits.max(dim=-1).values

    keep = scores >= conf_thresh
    scores = scores[keep]
    boxes = boxes[keep]

    if len(boxes) == 0:
        return np.empty((0, 4)), np.empty((0,))

    cx = boxes[:, 0] * orig_w
    cy = boxes[:, 1] * orig_h
    w = boxes[:, 2] * orig_w
    h = boxes[:, 3] * orig_h
    x1 = torch.clamp(cx - w / 2, min=0, max=orig_w)
    y1 = torch.clamp(cy - h / 2, min=0, max=orig_h)
    x2 = torch.clamp(cx + w / 2, min=0, max=orig_w)
    y2 = torch.clamp(cy + h / 2, min=0, max=orig_h)
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

    if nms_iou > 0 and len(boxes_xyxy) > 1:
        keep_idx = ops.nms(boxes_xyxy, scores, nms_iou)
        boxes_xyxy = boxes_xyxy[keep_idx]
        scores = scores[keep_idx]

    return boxes_xyxy.cpu().numpy(), scores.cpu().numpy()


def run_stage2_sam_filter(
    sam_model,
    img_input,
    candidate_boxes,
    scores,
    min_mask_area=40,
    mask_iou_thresh=0.50,
    device="cuda:2",
    chunk_size=None,
    prompt="car .",
    orig_img_shape=None,
):
    """Stage 2: Filter candidate boxes via SAM (SAM 2.1 Geometric or SAM 3 Semantic Foundation)."""
    # --------------------------------------------------------------------------
    # CASE A: SAM 3 Semantic Foundation Model Active (Vision-Language Prompting)
    # --------------------------------------------------------------------------
    if getattr(sam_model, "is_sam3_semantic", False):
        clean_prompt = prompt.rstrip(" .").strip()
        # Direct semantic visual-language grounding with SAM 3
        res = sam_model(source=str(img_input), text=clean_prompt)
        if len(res) == 0 or res[0].boxes is None or len(res[0].boxes) == 0:
            return np.empty((0, 4)), np.empty((0,))
        sem_boxes = res[0].boxes.xyxy.cpu().numpy()
        sem_scores = res[0].boxes.conf.cpu().numpy()
        return sem_boxes, sem_scores

    # --------------------------------------------------------------------------
    # CASE B: SAM 2.1 Geometric Mask Filtering + Physical Geometry Bounds
    # --------------------------------------------------------------------------
    if len(candidate_boxes) == 0:
        return np.empty((0, 4)), np.empty((0,))

    if chunk_size is None:
        chunk_size = 128

    boxes_list = candidate_boxes.tolist()
    masks_list = []
    for k in range(0, len(boxes_list), chunk_size):
        chunk = boxes_list[k:k + chunk_size]
        res = sam_model(img_input, bboxes=chunk, device=device, verbose=False)
        if len(res) > 0 and res[0].masks is not None:
            m = res[0].masks.data.cpu().numpy() > 0.5
            masks_list.append(m)

    if not masks_list:
        return np.empty((0, 4)), np.empty((0,))

    masks = np.concatenate(masks_list, axis=0)
    num_boxes = len(candidate_boxes)

    total_area = None
    if orig_img_shape is not None:
        total_area = orig_img_shape[0] * orig_img_shape[1]

    valid_candidates = []
    for i in range(num_boxes):
        box = candidate_boxes[i]
        score = scores[i]
        bw = box[2] - box[0]
        bh = box[3] - box[1]
        box_area = bw * bh

        # Physical Prior 1: Max Area bound (reject buildings/rooftops > 10% of image)
        if total_area is not None and (box_area / total_area) > 0.10:
            continue

        # Physical Prior 2: Aspect ratio bound (reject poles/cables)
        ar = bw / max(bh, 1.0)
        if ar < 0.20 or ar > 5.0:
            continue

        if i >= len(masks):
            continue

        mask_i = masks[i]
        mask_area = np.sum(mask_i)

        if mask_area < min_mask_area:
            continue

        fill_ratio = mask_area / max(box_area, 1.0)
        if fill_ratio < 0.04:
            continue

        valid_candidates.append({
            "box": box,
            "score": score,
            "mask": mask_i,
            "area": mask_area
        })

    valid_candidates.sort(key=lambda x: x["score"], reverse=True)
    kept_boxes = []
    kept_scores = []
    kept_masks = []
    kept_areas = []

    for cand in valid_candidates:
        box_cand = cand["box"]
        mask_cand = cand["mask"]
        area_cand = cand["area"]
        duplicate = False
        for k in range(len(kept_boxes)):
            box_k = kept_boxes[k]
            # Fast bounding-box disjoint rejection
            if box_cand[0] >= box_k[2] or box_cand[2] <= box_k[0] or box_cand[1] >= box_k[3] or box_cand[3] <= box_k[1]:
                continue
            ix1 = int(max(box_cand[0], box_k[0]))
            iy1 = int(max(box_cand[1], box_k[1]))
            ix2 = int(min(box_cand[2], box_k[2]))
            iy2 = int(min(box_cand[3], box_k[3]))
            intersection = np.sum(mask_cand[iy1:iy2, ix1:ix2] & kept_masks[k][iy1:iy2, ix1:ix2])
            union = area_cand + kept_areas[k] - intersection
            if (intersection / max(union, 1.0)) > mask_iou_thresh:
                duplicate = True
                break
        if not duplicate:
            kept_boxes.append(box_cand)
            kept_scores.append(cand["score"])
            kept_masks.append(mask_cand)
            kept_areas.append(area_cand)

    if len(kept_boxes) == 0:
        return np.empty((0, 4)), np.empty((0,))

    return np.array(kept_boxes), np.array(kept_scores)


def run_countvimg(model, transform, sam_model, img_path, device="cuda:2", prompt="car .", conf_thresh=0.20):
    pil_img = Image.open(img_path).convert("RGB")
    orig_shape = (pil_img.height, pil_img.width)

    # If SAM 3 Semantic Foundation Model is active:
    if sam_model is not None and getattr(sam_model, "is_sam3_semantic", False):
        final_boxes, final_scores = run_stage2_sam_filter(
            sam_model, str(img_path), None, None, device=device, prompt=prompt, orig_img_shape=orig_shape
        )
        return final_boxes, final_scores

    # Otherwise: CountGD-Box Stage 1 Detector + SAM 2.1 Stage 2 Filter
    cand_boxes, cand_scores = run_stage1_detector(
        model, transform, pil_img, device=device, prompt=prompt, conf_thresh=conf_thresh, nms_iou=0.70
    )
    if sam_model is not None:
        final_boxes, final_scores = run_stage2_sam_filter(
            sam_model, str(img_path), cand_boxes, cand_scores, device=device, prompt=prompt, orig_img_shape=orig_shape
        )
    else:
        final_boxes, final_scores = cand_boxes, cand_scores

    return final_boxes, final_scores


# ------------------------------------------------------------------------------
# 2. Evaluation Benchmarks
# ------------------------------------------------------------------------------
def resolve_path(p_str: str) -> Path:
    p = Path(p_str)
    if not p.is_absolute():
        p = ROOT / p
    elif not p.exists():
        # Check if it has legacy prefix
        parts = p.parts
        if "CS331" in parts:
            idx = parts.index("CS331")
            rel = Path(*parts[idx+1:])
            p = ROOT / rel
    return p


def eval_carpk_clean(model, transform, sam_model, device="cuda:2", max_samples=None):
    ann_file = ROOT / "data/carpk_occ_local_v1/clean.jsonl"
    print(f"\n[1/5] Evaluating CARPK Clean (Normal Counting) from: {ann_file.name}")
    records = []
    with open(ann_file) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if max_samples:
        records = records[:max_samples]

    errors = []
    for r in tqdm(records, desc="CARPK Clean"):
        img_path = resolve_path(r["image"])
        gt_count = float(r["count"])
        boxes, _ = run_countvimg(model, transform, sam_model, img_path, device=device, prompt="car .")
        pred_count = float(len(boxes))
        errors.append(pred_count - gt_count)

    err = np.array(errors)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    print(f"--> CARPK Clean Results: Images={len(err)} | MAE = {mae:.2f} | RMSE = {rmse:.2f}")
    return {"images": len(err), "MAE": round(mae, 2), "RMSE": round(rmse, 2)}


def eval_carpk_occ_levels(model, transform, sam_model, device="cuda:2", max_samples=None):
    print("\n[2/5] Evaluating CARPK-OCC-LOCAL (Stratified Occlusion: Mild, Moderate, Heavy)...")
    levels = ["mild", "moderate", "heavy"]
    results = {}

    for lvl in levels:
        ann_file = ROOT / f"data/carpk_occ_local_v1/{lvl}.jsonl"
        records = []
        with open(ann_file) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        if max_samples:
            records = records[:max_samples]

        errors = []
        for r in tqdm(records, desc=f"CARPK-OCC {lvl.capitalize()}"):
            img_path = resolve_path(r["image"])
            gt_count = float(r["count"])
            boxes, _ = run_countvimg(model, transform, sam_model, img_path, device=device, prompt="car .")
            pred_count = float(len(boxes))
            errors.append(pred_count - gt_count)

        err = np.array(errors)
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        results[lvl] = {"images": len(err), "MAE": round(mae, 2), "RMSE": round(rmse, 2)}
        print(f"--> CARPK-OCC {lvl.capitalize()}: MAE = {mae:.2f} | RMSE = {rmse:.2f}")

    return results


def eval_trancos(model, transform, sam_model, device="cuda:2", max_samples=None):
    ann_file = ROOT / "data/trancos_test_eval.json"
    print(f"\n[3/5] Evaluating TRANCOS (Crowded Traffic Counting & Localization)...")
    with open(ann_file) as f:
        records = json.load(f)
    if max_samples:
        records = records[:max_samples]

    errors = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for r in tqdm(records, desc="TRANCOS"):
        img_path = resolve_path(r["image_path"])
        gt_count = float(r["gt_count"])
        gt_boxes = np.array(r.get("gt_boxes", []))

        pred_boxes, _ = run_countvimg(model, transform, sam_model, img_path, device=device, prompt="car .")
        pred_count = float(len(pred_boxes))
        errors.append(pred_count - gt_count)

        # Match predicted boxes to GT boxes (IoU >= 0.5)
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            ious = ops.box_iou(torch.tensor(pred_boxes, dtype=torch.float32), torch.tensor(gt_boxes, dtype=torch.float32)).numpy()
            matched_gt = set()
            tp = 0
            for i in range(len(pred_boxes)):
                max_j = int(np.argmax(ious[i]))
                if ious[i, max_j] >= 0.50 and max_j not in matched_gt:
                    matched_gt.add(max_j)
                    tp += 1
            fp = len(pred_boxes) - tp
            fn = len(gt_boxes) - len(matched_gt)
        else:
            tp = 0
            fp = len(pred_boxes)
            fn = len(gt_boxes)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    err = np.array(errors)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)
    f1 = (2 * precision * recall) / max(1e-6, precision + recall)

    print(f"--> TRANCOS Results: MAE = {mae:.2f} | RMSE = {rmse:.2f} | Precision = {precision*100:.1f}% | Recall = {recall*100:.1f}% | F1 = {f1*100:.1f}%")
    return {
        "images": len(err),
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "Precision": round(precision * 100.0, 1),
        "Recall": round(recall * 100.0, 1),
        "F1": round(f1 * 100.0, 1)
    }


def eval_ovad_color(model, transform, sam_model, device="cuda:2", num_images=100):
    print(f"\n[4/5] Evaluating OVAD Attribute-Guided Counting ('red car')...")
    d = ROOT / "data/semantic_external_v1/ovad_expert"
    labels = {json.loads(l)["id"]: json.loads(l) for l in open(d / "labels.jsonl")}
    requests = [json.loads(l) for l in open(d / "requests.jsonl")]

    # Common 8 colors in OVAD: ['black', 'blue', 'brown', 'gray', 'green', 'red', 'white', 'yellow']
    # 'red' is index 5
    red_index = 5
    vehicle_reqs = [r for r in requests if r.get("domain") == "vehicle"]

    # Group by image
    img_map = {}
    for r in vehicle_reqs:
        img_id = r["image_id"]
        if img_id not in img_map:
            img_map[img_id] = {"image": r["image"], "red_boxes": []}
        req_id = r["id"]
        lbl = labels.get(req_id, {}).get("target", [])
        if len(lbl) > red_index and lbl[red_index] == 1:
            img_map[img_id]["red_boxes"].append(r["box"])

    eval_items = list(img_map.items())[:num_images]

    total_tp = 0
    total_fp = 0
    total_fn = 0
    count_errors = []

    for img_id, item in tqdm(eval_items, desc="OVAD Red Car"):
        img_path = resolve_path(item["image"])
        gt_boxes = np.array(item["red_boxes"])
        gt_count = float(len(gt_boxes))

        pred_boxes, _ = run_countvimg(model, transform, sam_model, img_path, device=device, prompt="red car .")
        pred_count = float(len(pred_boxes))
        count_errors.append(abs(pred_count - gt_count))

        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            ious = ops.box_iou(torch.tensor(pred_boxes, dtype=torch.float32), torch.tensor(gt_boxes, dtype=torch.float32)).numpy()
            matched_gt = set()
            tp = 0
            for i in range(len(pred_boxes)):
                max_j = int(np.argmax(ious[i]))
                if ious[i, max_j] >= 0.50 and max_j not in matched_gt:
                    matched_gt.add(max_j)
                    tp += 1
            fp = len(pred_boxes) - tp
            fn = len(gt_boxes) - len(matched_gt)
        else:
            tp = 0
            fp = len(pred_boxes)
            fn = len(gt_boxes)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)
    f1 = (2 * precision * recall) / max(1e-6, precision + recall)
    count_mae = float(np.mean(count_errors))

    print(f"--> OVAD Results: Images={len(eval_items)} | Localization F1 = {f1*100:.1f}% | Count MAE = {count_mae:.2f}")
    return {
        "images": len(eval_items),
        "Localization_F1": round(f1 * 100.0, 1),
        "Count_MAE": round(count_mae, 2)
    }


def eval_hico_relation(model, transform, sam_model, device="cuda:2", num_images=100):
    print(f"\n[5/5] Evaluating HICO Interaction Recognition ('person riding motorcycle')...")
    ann_file = ROOT / "data/semantic_external_v1/hico_expert/images.json"
    with open(ann_file) as f:
        imgs = json.load(f)

    test_imgs = [img for img in imgs if img.get("role") == "test"][:num_images]

    gt_presences = []
    pred_presences = []
    count_errors = []

    for r in tqdm(test_imgs, desc="HICO Relation"):
        img_path = resolve_path(r["image"])
        gt_presence = 1 if r.get("raw_image_onboard", 0) == 1 else 0
        gt_count = float(r.get("eligible_onboard_count", 0))

        pred_boxes, _ = run_countvimg(model, transform, sam_model, img_path, device=device, prompt="motorcycle with a person riding it .")
        pred_count = float(len(pred_boxes))
        pred_presence = 1 if pred_count > 0 else 0

        gt_presences.append(gt_presence)
        pred_presences.append(pred_presence)
        count_errors.append(abs(pred_count - gt_count))

    gt_p = np.array(gt_presences)
    pr_p = np.array(pred_presences)

    tp = np.sum((gt_p == 1) & (pr_p == 1))
    fp = np.sum((gt_p == 0) & (pr_p == 1))
    fn = np.sum((gt_p == 1) & (pr_p == 0))

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall) / max(1e-6, precision + recall)
    count_mae = float(np.mean(count_errors))

    print(f"--> HICO Results: Images={len(test_imgs)} | Precision = {precision*100:.1f}% | Recall = {recall*100:.1f}% | F1 = {f1*100:.1f}% | Count MAE = {count_mae:.2f}")
    return {
        "images": len(test_imgs),
        "Precision": round(precision * 100.0, 1),
        "Recall": round(recall * 100.0, 1),
        "F1": round(f1 * 100.0, 1),
        "Count_MAE": round(count_mae, 2)
    }


# ------------------------------------------------------------------------------
# 3. Main Benchmark Runner
# ------------------------------------------------------------------------------
def run_full_suite_for_model(model_name: str, ckpt_path: Path, sam_model, device: str = "cuda:2", quick_test: bool = False, master_results: dict = None, out_file: Path = None):
    print("=" * 80)
    print(f" STARTING BENCHMARK FOR MODEL: {model_name}")
    print(f" Checkpoint: {ckpt_path.name}")
    print("=" * 80)

    if master_results is None:
        master_results = {}
    if model_name not in master_results:
        master_results[model_name] = {
            "model_name": model_name,
            "checkpoint": ckpt_path.name,
            "elapsed_seconds": 0.0
        }
    entry = master_results[model_name]

    def save_progress():
        if out_file:
            with open(out_file, "w") as f:
                json.dump(master_results, f, indent=2)

    model, transform = load_detector_model(ckpt_path, device=device)

    max_samples = 3 if quick_test else None
    num_sem_images = 3 if quick_test else 100
    expected_carpk = 3 if quick_test else 459
    expected_trancos = 3 if quick_test else 421
    expected_sem = 3 if quick_test else 100

    t0 = time.time()

    # 1. CARPK Clean
    if "CARPK_clean" not in entry or entry["CARPK_clean"].get("images", 0) < expected_carpk:
        entry["CARPK_clean"] = eval_carpk_clean(model, transform, sam_model, device=device, max_samples=max_samples)
        save_progress()
    else:
        print(f"[*] CARPK Clean already completed ({entry['CARPK_clean']}) -> Skipping.")

    # 2. CARPK-OCC Levels
    need_occ = False
    if "CARPK_OCC" not in entry or not isinstance(entry["CARPK_OCC"], dict):
        need_occ = True
    else:
        for lvl in ["mild", "moderate", "heavy"]:
            if lvl not in entry["CARPK_OCC"] or entry["CARPK_OCC"][lvl].get("images", 0) < expected_carpk:
                need_occ = True
                break
    if need_occ:
        entry["CARPK_OCC"] = eval_carpk_occ_levels(model, transform, sam_model, device=device, max_samples=max_samples)
        save_progress()
    else:
        print(f"[*] CARPK-OCC already completed -> Skipping.")

    # 3. TRANCOS
    if "TRANCOS" not in entry or entry["TRANCOS"].get("images", 0) < expected_trancos:
        entry["TRANCOS"] = eval_trancos(model, transform, sam_model, device=device, max_samples=max_samples)
        save_progress()
    else:
        print(f"[*] TRANCOS already completed -> Skipping.")

    # 4. OVAD
    if "OVAD" not in entry or entry["OVAD"].get("images", 0) < expected_sem:
        entry["OVAD"] = eval_ovad_color(model, transform, sam_model, device=device, num_images=num_sem_images)
        save_progress()
    else:
        print(f"[*] OVAD already completed -> Skipping.")

    # 5. HICO
    if "HICO" not in entry or entry["HICO"].get("images", 0) < expected_sem:
        entry["HICO"] = eval_hico_relation(model, transform, sam_model, device=device, num_images=num_sem_images)
        save_progress()
    else:
        print(f"[*] HICO already completed -> Skipping.")

    elapsed = time.time() - t0
    entry["elapsed_seconds"] = round(entry.get("elapsed_seconds", 0.0) + elapsed, 1)
    save_progress()

    # Cleanup GPU memory
    del model
    del transform
    torch.cuda.empty_cache()

    return entry


def main():
    parser = argparse.ArgumentParser(description="Full Benchmark Suite for CountVImg on 5 Datasets")
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--quick-test", action="store_true", help="Quick test run on small subset")
    parser.add_argument("--model-idx", type=int, default=None, choices=[0, 1, 2], help="Run only specific model (0, 1, or 2)")
    parser.add_argument("--sam-checkpoint", type=str, default=str(ROOT / "Cubi/sam2.1_t.pt"), help="Path to SAM checkpoint")
    parser.add_argument("--out-json", type=str, default=str(ROOT / "outputs/countvimg_evaluation/master_benchmark_results.json"))
    args = parser.parse_args()

    out_file = Path(args.out_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    sam_path = Path(args.sam_checkpoint)
    if "sam3" in sam_path.stem:
        sam_tag = "SAM 3"
        print(f"[*] Initializing SAM 3 Semantic Foundation Model from {sam_path.name} on {args.device}...")
        from ultralytics.models.sam.predict import SAM3SemanticPredictor
        overrides = dict(conf=0.20, task="segment", mode="predict", model=str(sam_path), device=args.device, verbose=False, save=False)
        sam_model = SAM3SemanticPredictor(overrides=overrides)
        sam_model.setup_model(model=None, verbose=False)
        sam_model.is_sam3_semantic = True
    else:
        sam_tag = "SAM 2.1"
        print(f"[*] Initializing SAM 2.1 from {sam_path.name} on {args.device}...")
        sam_model = SAM(str(sam_path))
        sam_model.is_sam3_semantic = False

    models_to_test = [
        (f"Baseline (CountGD-Box Gốc + {sam_tag})", ROOT / "checkpoints/countgd_box.pth"),
        (f"Cải tiến 1 (VisDrone AP-Tuned + {sam_tag})", ROOT / "outputs/countgd_box_ap_optimized/checkpoint_best_ap.pth"),
        (f"Cải tiến 2 (Occlusion-Trained + {sam_tag})", ROOT / "outputs/countgd_box_occ_step2/checkpoint_best.pth")
    ]

    if args.model_idx is not None:
        models_to_test = [models_to_test[args.model_idx]]

    master_results = {}
    if out_file.exists():
        try:
            with open(out_file) as f:
                master_results = json.load(f)
        except Exception:
            master_results = {}

    for name, path in models_to_test:
        res = run_full_suite_for_model(
            name, path, sam_model, device=args.device, quick_test=args.quick_test, master_results=master_results, out_file=out_file
        )
        master_results[name] = res
        with open(out_file, "w") as f:
            json.dump(master_results, f, indent=2)
        print(f"[+] Saved updated results to {out_file.name}")

    print("\n" + "=" * 80)
    print(" ALL BENCHMARKS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
