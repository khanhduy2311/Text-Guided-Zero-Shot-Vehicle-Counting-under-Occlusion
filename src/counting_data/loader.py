"""Strict manifest loader; does not use legacy mock-image or label fallbacks.

Returns an image tensor and an explicit target dictionary. It is a data API, not
a drop-in replacement for legacy CountGD ODVG, which discards ignore/point data.
"""
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from scipy.io import loadmat
from torch.utils.data import Dataset

from .schema import SOURCES, TARGETS, selected_instances, validate_record


class UnifiedCountingDataset(Dataset):
    def __init__(self, manifest, root, *, purpose, query=None, short_side=None,
                 max_size=1333, horizontal_flip_p=0.0, normalize=True,
                 text_only=True, verify_schema=True):
        if purpose not in {'train', 'validate', 'evaluate'}:
            raise ValueError('purpose must be train, validate, or evaluate')
        if not 0 <= horizontal_flip_p <= 1:
            raise ValueError('Invalid flip probability')
        if purpose != 'train' and horizontal_flip_p:
            raise ValueError('Evaluation augmentation must be explicitly implemented separately')
        if short_side is not None and short_side <= 0:
            raise ValueError('short_side must be positive')
        if max_size <= 0:
            raise ValueError('max_size must be positive')
        self.root = Path(root).resolve()
        self.manifest = Path(manifest)
        self.query, self.purpose = query, purpose
        self.short_side, self.max_size = short_side, max_size
        self.flip_p, self.normalize, self.text_only = horizontal_flip_p, normalize, text_only
        self.offsets = []
        allowed = {'train': {'source_train'}, 'validate': {'source_val'},
                   'evaluate': {'source_val', 'source_test', 'target_test'}}[purpose]
        # Index offsets, not full dictionaries; large JSONL files remain worker-friendly.
        with self.manifest.open('rb') as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                r = json.loads(line)
                if r['role'] not in allowed:
                    raise ValueError(f"{purpose} forbids role {r['role']} ({r['id']})")
                if purpose != 'evaluate' and (r['dataset'] not in SOURCES or r['dataset'] in TARGETS):
                    raise ValueError('Protected or unapproved dataset in source manifest')
                if verify_schema:
                    validate_record(r)
                selected_instances(r, query)
                self.offsets.append(offset)

    def __len__(self):
        return len(self.offsets)

    def record(self, index):
        with self.manifest.open('rb') as f:
            f.seek(self.offsets[index])
            return json.loads(f.readline())

    def __getitem__(self, index):
        r = self.record(index)
        path = (self.root / r['image']).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Image path escapes workspace')
        # Missing or corrupt images raise; never substitute a gray mock image.
        with Image.open(path) as im:
            image = im.convert('RGB')
        w, h = image.size
        if (w, h) != (r['width'], r['height']):
            raise ValueError(f"Stored dimensions differ: {r['id']}")
        instances = copy.deepcopy(selected_instances(r, self.query))
        nw, nh = w, h
        if self.short_side is not None:
            scale = min(self.short_side / min(w, h), self.max_size / max(w, h))
            nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
            image = image.resize((nw, nh), Image.Resampling.BILINEAR)
        flip = bool(self.flip_p and torch.rand(()).item() < self.flip_p)
        if flip:
            image = ImageOps.mirror(image)
        sx, sy = nw / w, nh / h

        def point(p):
            x, y = p[0] * sx, p[1] * sy
            return [nw - x if flip else x, y]

        def box(b):
            x1, y1, x2, y2 = b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy
            return [nw - x2, y1, nw - x1, y2] if flip else [x1, y1, x2, y2]

        points = torch.tensor([point(i['point']) for i in instances], dtype=torch.float32).reshape(-1, 2)
        box_indices = [j for j, i in enumerate(instances) if i['box'] is not None]
        boxes = torch.tensor([box(instances[j]['box']) for j in box_indices], dtype=torch.float32).reshape(-1, 4)
        ignore_regions = copy.deepcopy(r['ignore_regions'])
        for region in ignore_regions:
            region['box'] = box(region['box'])
        ignore_boxes = torch.tensor([v['box'] for v in ignore_regions], dtype=torch.float32).reshape(-1, 4)
        exemplars = [] if self.text_only else [box(b) for b in r['exemplar_boxes']]
        occlusion = []
        for i in instances:
            occ = copy.deepcopy(i.get('occlusion'))
            if occ is not None:
                for region in occ.get('regions', []):
                    region['box'] = box(region['box'])
            occlusion.append(occ)
        roi_mask = torch.ones((nh, nw), dtype=torch.bool)
        if r['roi'] is not None:
            roi = r['roi']
            roi_path = (self.root / roi['path']).resolve()
            if not roi_path.is_relative_to(self.root):
                raise ValueError('ROI path escapes workspace')
            mask = np.asarray(loadmat(roi_path)[roi['key']])
            if mask.shape != (h, w):
                raise ValueError('ROI dimensions differ from image')
            mask_image = Image.fromarray((mask != 0).astype(np.uint8) * 255)
            mask_image = mask_image.resize((nw, nh), Image.Resampling.NEAREST)
            if flip:
                mask_image = ImageOps.mirror(mask_image)
            roi_mask = torch.from_numpy(np.array(mask_image) != 0)
        supervision_mask = roi_mask.clone()
        for x1, y1, x2, y2 in ignore_boxes.tolist():
            supervision_mask[math.floor(y1):math.ceil(y2), math.floor(x1):math.ceil(x2)] = False
        tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255.0
        if self.normalize:
            mean = tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]
            std = tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
            tensor = (tensor - mean) / std
        return tensor, {
            'id': r['id'], 'dataset': r['dataset'], 'role': r['role'],
            'scene_id': r['scene_id'], 'group_id': r['group_id'],
            'prompt': self.query if self.query is not None else r['prompt'],
            'count': len(instances), 'supervision': r['supervision'],
            'instance_ids': [i['id'] for i in instances], 'classes': [i['class'] for i in instances],
            'points_xy': points, 'points_normalized': points / points.new_tensor([nw, nh]),
            'point_origins': [i['point_origin'] for i in instances],
            'boxes_xyxy': boxes, 'boxes_normalized_xyxy': boxes / boxes.new_tensor([nw, nh, nw, nh]),
            'box_instance_indices': torch.tensor(box_indices, dtype=torch.int64),
            'exemplar_boxes_xyxy': torch.tensor(exemplars, dtype=torch.float32).reshape(-1, 4),
            'ignore_boxes_xyxy': ignore_boxes, 'ignore_regions': ignore_regions,
            'occlusion': occlusion, 'truncation': [i.get('truncation') for i in instances],
            'size_hw': torch.tensor([nh, nw]), 'original_size_hw': torch.tensor([h, w]),
            'scale_xy': (sx, sy), 'flipped': flip,
            # Targets for loss/evaluation ONLY. Never inputs to image/text inference.
            'roi_mask': roi_mask, 'supervision_mask': supervision_mask,
            'annotation_ref': r['annotation_ref'],
        }


def counting_collate(batch):
    """Keep variable image sizes and instance counts; the model adapter handles padding."""
    images, targets = zip(*batch)
    return list(images), list(targets)
