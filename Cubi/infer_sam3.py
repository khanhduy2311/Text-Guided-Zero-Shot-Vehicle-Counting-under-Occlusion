"""Local CLI for the two-stage CountGD-Box + SAM 3 inference used by CountLab."""
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    proposals = stages.add_parser("proposals")
    proposals.add_argument("--checkpoint", type=Path, required=True)
    proposals.add_argument("--noun", required=True)
    proposals.add_argument("--out", type=Path, required=True)
    verify = stages.add_parser("verify")
    verify.add_argument("--sam3", required=True)
    verify.add_argument("--prompt", required=True)
    verify.add_argument("--proposals", type=Path, required=True)
    for stage in (proposals, verify):
        stage.add_argument("--image", type=Path, required=True)
        stage.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    import hashlib
    import torch
    from PIL import Image
    identity = hashlib.sha256(args.image.read_bytes()).hexdigest()
    with Image.open(args.image) as source, torch.inference_mode():
        image = source.convert("RGB")
        if args.stage == "proposals":
            from Cubi.benchmark_countvimg_suite import load_detector_model, run_stage1_detector
            model, transform = load_detector_model(args.checkpoint, args.device)
            boxes, scores = run_stage1_detector(model, transform, image, args.device, args.noun)
            args.out.write_text(json.dumps(dict(image_sha256=identity, boxes=boxes.tolist(), scores=scores.tolist())))
            return
        from src.semantic_benchmark_metrics import match
        from transformers import Sam3Processor, Sam3VideoModel
        candidates = json.loads(args.proposals.read_text())
        if candidates["image_sha256"] != identity:
            parser.error("Proposals refer to a different source image")
        processor = Sam3Processor.from_pretrained(args.sam3, local_files_only=True)
        model = Sam3VideoModel.from_pretrained(args.sam3, local_files_only=True, torch_dtype=torch.float32)
        detector = model.detector_model.eval().to(args.device)
        inputs = processor(images=image, text=args.prompt, return_tensors="pt").to(args.device)
        instances = processor.image_processor.post_process_object_detection(
            detector(**inputs), threshold=0.20, target_sizes=[image.size[::-1]])[0]
        semantic_boxes = instances["boxes"].float().cpu().tolist()
        scores = instances["scores"].float().cpu().tolist()
        boxes = candidates["boxes"]
        objects = [dict(box=boxes[i], detector_score=candidates["scores"][i], semantic_score=scores[j], match_iou=iou)
                   for i, j, iou in match(boxes, semantic_boxes, 0.30)]
        print(json.dumps(dict(prompt=args.prompt, count=len(objects), objects=objects)))

if __name__ == "__main__":
    main()
