"""A2 count-preserving synthetic occlusion on annotated source vehicle boxes.

Masks are augmentation metadata, never model inputs. Bounding-box visible-area
constraints are proxies, not proof of visible object pixels. Point-only FSC147
is deliberately left unchanged because instance extent is unknown.
"""
import hashlib
import math
import random

import numpy as np
import torch

from .counting_eval import occlusion_bin


def deterministic_seed(seed, image_id, epoch=0, offset=0, mode='train'):
    value = f'{seed}|{image_id}|{epoch}|{offset}|{mode}'.encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], 'big')


def box_mask_fractions(mask, boxes):
    """Pixel-area union coverage, including fractional/small-box boundaries."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if not len(boxes):
        return np.empty(0, dtype=np.float64)
    h, w = mask.shape
    x1 = np.clip(np.floor(boxes[:, 0]).astype(int), 0, w-1)
    y1 = np.clip(np.floor(boxes[:, 1]).astype(int), 0, h-1)
    x2 = np.maximum(x1+1, np.clip(np.ceil(boxes[:, 2]).astype(int), 0, w))
    y2 = np.maximum(y1+1, np.clip(np.ceil(boxes[:, 3]).astype(int), 0, h))
    integral = np.pad(mask.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    covered = integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
    return covered / ((x2-x1)*(y2-y1))


def occlude_source(image, target, policy, *, epoch, offset, mode='train', force=False):
    if mode not in ('train', 'validation'):
        raise ValueError('Unknown augmentation mode')
    role = 'source_train' if mode == 'train' else 'source_val'
    if target['role'] != role or target['dataset'] not in ('FSC147', 'VisDrone', 'UA_DETRAC'):
        raise ValueError('Synthetic occlusion is restricted to approved source roles')
    if image.device.type != 'cpu' or image.ndim != 3 or image.shape[0] != 3:
        raise ValueError('Augment normalized RGB CHW on CPU before model forward')
    h, w = image.shape[-2:]
    mask = np.zeros((h, w), dtype=np.bool_)
    meta = {'applied': False, 'mode': mode, 'id': target['id'], 'count_preserved': target['count'],
            'mask_fraction': 0., 'max_added_box_fraction': 0., 'occluder_type': None}
    if target['supervision'] != 'box':
        meta['skip_reason'] = 'point_only_extent_unknown'
        return image, torch.from_numpy(mask), meta
    if mode == 'train':
        stage = policy['curriculum'][min(epoch, len(policy['curriculum'])-1)]
        seed = deterministic_seed(policy['seed'], target['id'], epoch, offset, mode)
        styles = ('vertical_pole', 'horizontal_barrier')
    else:
        stage = policy['fixed_validation']
        seed = deterministic_seed(policy['seed'], target['id'], mode=mode)
        styles = ('diagonal_band',)  # held-out geometry, never generated in training
    rng = random.Random(seed)
    meta.update(seed=seed, epoch_stage=epoch+1 if mode == 'train' else None)
    if not force and rng.random() >= stage['probability']:
        meta['skip_reason'] = 'probability'
        return image, torch.from_numpy(mask), meta
    boxes = target['boxes_xyxy'].numpy()
    limits = np.full(len(boxes), stage['max_added_box_fraction'], dtype=np.float64)
    for i, box in enumerate(boxes):
        severity = occlusion_bin(target['occlusion'][i], box)
        if severity == 'heavy': limits[i] = 0.
        elif severity == 'partial': limits[i] = min(limits[i], .20)
        elif severity == 'unknown': limits[i] = min(limits[i], .10)
    anchors = np.flatnonzero(limits >= .15).tolist()
    accepted = 0
    for _ in range(policy['max_attempts']):
        style = rng.choice(styles)
        if anchors:
            x1, y1, x2, y2 = boxes[rng.choice(anchors)]
            bw, bh = x2-x1, y2-y1
            cx = rng.uniform(x1+.2*bw, x2-.2*bw)
            cy = rng.uniform(y1+.2*bh, y2-.2*bh)
        elif len(boxes):
            meta['skip_reason'] = 'no_safe_anchor'
            break
        else:
            bw, bh = w*.15, h*.15
            cx, cy = rng.uniform(.1*w, .9*w), rng.uniform(.1*h, .9*h)
        thickness = rng.uniform(.08, min(.25, stage['max_added_box_fraction']))
        if style == 'vertical_pole':
            rw, rh = max(1., bw*thickness), bh*rng.uniform(1.1, 2.)
        elif style == 'horizontal_barrier':
            rw, rh = bw*rng.uniform(1.1, 2.), max(1., bh*thickness)
        else:
            rw, rh = bw*1.4, bh*1.4
        left, right = max(0, math.floor(cx-rw/2)), min(w, math.ceil(cx+rw/2))
        top, bottom = max(0, math.floor(cy-rh/2)), min(h, math.ceil(cy+rh/2))
        candidate = mask.copy()
        if style == 'diagonal_band':
            yy, xx = np.mgrid[top:bottom, left:right]
            band = np.abs((xx-cx) + (yy-cy)*rng.choice((.6, -.6))) <= max(1., bw*thickness/2)
            candidate[top:bottom, left:right] |= band
        else:
            candidate[top:bottom, left:right] = True
        fractions = box_mask_fractions(candidate, boxes)
        if candidate.mean() > stage['max_image_fraction'] or np.any(fractions > limits + 1e-12):
            continue
        if np.array_equal(candidate, mask):
            continue
        mask = candidate; accepted += 1
        meta['occluder_type'] = style
        if accepted >= stage['occluders']:
            break
    if not mask.any():
        meta.setdefault('skip_reason', 'visibility_constraints')
        return image, torch.from_numpy(mask), meta
    # Neutral synthetic material, no vehicle copy-paste or change in instance count.
    grey = rng.uniform(.12, .75)
    rgb = image.new_tensor([min(1., max(0., grey+rng.uniform(-.03, .03))) for _ in range(3)])
    mean = image.new_tensor([.485, .456, .406]); std = image.new_tensor([.229, .224, .225])
    result = image.clone(); tensor_mask = torch.from_numpy(mask)
    result[:, tensor_mask] = ((rgb-mean)/std)[:, None]
    fractions = box_mask_fractions(mask, boxes)
    meta.update(applied=True, accepted_occluders=accepted, mask_fraction=float(mask.mean()),
                max_added_box_fraction=float(fractions.max()) if len(fractions) else 0.,
                mask_sha256=hashlib.sha256(np.packbits(mask).tobytes()).hexdigest(),
                bbox_visibility_proxy_only=True)
    return result, tensor_mask, meta
