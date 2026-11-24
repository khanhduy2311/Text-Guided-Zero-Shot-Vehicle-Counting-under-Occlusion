"""CPU metrics for the frozen source protocol; masks never affect inference.

Box localization: maximum-cardinality one-to-one matching at IoU >= 0.5,
with IoU breaking ties. Point localization: distance <= 2% image diagonal
(project diagnostic, not an official FSC147 localization metric).
ROI excludes predictions first. Annotated positives take precedence over ignore
regions: after matching, unmatched predictions centered in ignore are omitted.
"""
import math
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment


def box_iou(a, b):
    a, b = np.asarray(a).reshape(-1, 4), np.asarray(b).reshape(-1, 4)
    inter = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                       np.maximum(a[:, None, :2], b[None, :, :2])).prod(-1)
    aa = np.maximum(0, a[:, 2:]-a[:, :2]).prod(-1)
    bb = np.maximum(0, b[:, 2:]-b[:, :2]).prod(-1)
    return inter / np.maximum(aa[:, None]+bb[None, :]-inter, 1e-12)


def match_pairs(quality, valid):
    if not quality.shape[0] or not quality.shape[1]:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    # One extra valid pair outweighs every quality tie-break combined.
    cost = np.where(valid, -1-quality/(min(quality.shape)+1), 0)
    rows, cols = linear_sum_assignment(cost)
    keep = valid[rows, cols]
    return rows[keep], cols[keep]


def union_area(boxes):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if not len(boxes):
        return 0.
    total = 0.
    xs = sorted(set(boxes[:, [0, 2]].ravel()))
    for lo, hi in zip(xs[:-1], xs[1:]):
        intervals = sorted((b[1], b[3]) for b in boxes if b[0] < hi and b[2] > lo and b[3] > b[1])
        covered, end = 0., -math.inf
        for start, stop in intervals:
            covered += max(0., stop-max(start, end))
            end = max(end, stop)
        total += (hi-lo)*covered
    return total


def occlusion_bin(occ, box):
    if occ is None:
        return 'unknown'
    if occ['dataset'] == 'VisDrone':
        return {0: 'clean', 1: 'partial', 2: 'heavy'}.get(occ.get('code'), 'unknown')
    if occ['dataset'] == 'UA_DETRAC':
        if not occ.get('present'):
            return 'clean'
        regions = occ.get('regions', [])
        if not regions or box is None:
            return 'unknown'
        clipped = []
        for region in regions:
            b = region['box']
            clipped.append([max(b[0], box[0]), max(b[1], box[1]), min(b[2], box[2]), min(b[3], box[3])])
        clipped = [b for b in clipped if b[2] > b[0] and b[3] > b[1]]
        ratio = union_area(clipped)/max((box[2]-box[0])*(box[3]-box[1]), 1e-12)
        return 'clean' if ratio == 0 else ('partial' if ratio < .5 else 'heavy')
    return 'unknown'


def density_bin(count):
    for upper, name in [(0, '0'), (10, '1-10'), (50, '11-50'), (100, '51-100'), (900, '101-900')]:
        if count <= upper:
            return name
    return '>900'


def evaluate_image(prediction, target, *, threshold=.23, text_threshold=0., box_iou_threshold=.5,
                   point_radius_fraction=.02):
    scores = np.asarray(prediction['scores'])
    boxes_c = np.asarray(prediction['boxes'])
    text_min = np.asarray(prediction['text_min'])
    if not np.isfinite(scores).all() or not np.isfinite(boxes_c).all() or not np.isfinite(text_min).all():
        raise ValueError('Nonfinite prediction')
    keep = (scores > threshold) & (text_min > text_threshold)
    query_indices = np.flatnonzero(keep)
    boxes_c = boxes_c[keep]
    h, w = target['size_hw'].tolist()
    oh, ow = target['original_size_hw'].tolist()
    points = boxes_c[:, :2] * [w, h]
    xyxy = np.concatenate((boxes_c[:, :2]-boxes_c[:, 2:]/2, boxes_c[:, :2]+boxes_c[:, 2:]/2), -1)
    xyxy *= [w, h, w, h]
    xyxy[:, 0::2] = np.clip(xyxy[:, 0::2], 0, w)
    xyxy[:, 1::2] = np.clip(xyxy[:, 1::2], 0, h)
    if target['flipped']:
        raise ValueError('Evaluator forbids flipped validation images')
    roi = target['roi_mask'].numpy()
    def mask_at(mask, pts):
        ix = np.clip(np.floor(pts[:, 0]).astype(int), 0, w-1)
        iy = np.clip(np.floor(pts[:, 1]).astype(int), 0, h-1)
        return mask[iy, ix]
    gt_points = target['points_xy'].numpy()
    if not mask_at(roi, gt_points).all():
        raise ValueError('Annotated GT outside ROI; define benchmark policy before evaluation')
    inside_roi = mask_at(roi, points)
    candidates = np.flatnonzero(inside_roi)
    if target['supervision'] == 'box':
        gt_boxes = target['boxes_xyxy'].numpy()
        quality = box_iou(xyxy[candidates], gt_boxes)
        valid = quality >= box_iou_threshold
        loc_metric = 'box_iou_0.5'
    else:
        gt_boxes = None
        distance = np.linalg.norm(points[candidates, None, :]-gt_points[None, :, :], axis=-1)
        radius = point_radius_fraction*math.hypot(w, h)
        quality = np.clip(1-distance/radius, 0, 1)
        valid = distance <= radius
        loc_metric = 'point_radius_0.02_diagonal_project_diagnostic'
    rows, cols = match_pairs(quality, valid)
    matched = candidates[rows]
    annotated = np.zeros(len(points), dtype=bool)
    annotated[matched] = True
    # Exact continuous ignore rectangle membership; no resize rounding artifacts.
    in_ignore = np.zeros(len(points), dtype=bool)
    for x1, y1, x2, y2 in target['ignore_boxes_xyxy'].tolist():
        in_ignore |= ((points[:, 0] >= x1) & (points[:, 0] < x2) &
                      (points[:, 1] >= y1) & (points[:, 1] < y2))
    ignored = inside_roi & in_ignore & ~annotated
    accepted = inside_roi & ~ignored
    gt_count = int(target['count'])
    if gt_count != len(gt_points):
        raise ValueError('GT count mismatch')
    tp = len(matched)
    count = int(accepted.sum())
    bins, sizes, classes = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    found = set(cols.tolist())
    occ_names = []
    sx, sy = target['scale_xy']
    for i in range(gt_count):
        box = None if gt_boxes is None else gt_boxes[i]
        occ = occlusion_bin(target['occlusion'][i], box)
        occ_names.append(occ)
        if box is None:
            size = 'unknown'
        else:
            area = (box[2]-box[0])*(box[3]-box[1])/(sx*sy)
            size = 'small_<32px' if area < 32**2 else ('medium_32-96px' if area < 96**2 else 'large_>=96px')
        for mapping, key in [(bins, occ), (sizes, size), (classes, target['classes'][i])]:
            mapping[key][0] += 1
            mapping[key][1] += int(i in found)
    severity = {'clean': 0, 'partial': 1, 'heavy': 2, 'unknown': -1}
    image_occ = max(occ_names, key=lambda x: severity[x]) if occ_names else 'empty'
    points_orig = points/np.array([sx, sy])
    boxes_orig = xyxy/np.array([sx, sy, sx, sy])
    return {'threshold': threshold, 'raw_count': int(keep.sum()), 'pred_count': count,
            'gt_count': gt_count, 'error': count-gt_count, 'tp': tp, 'fp': count-tp, 'fn': gt_count-tp,
            'outside_roi_count': int((~inside_roi).sum()), 'ignored_unmatched_count': int(ignored.sum()),
            'localization_metric': loc_metric, 'recall_by_occlusion': dict(bins), 'recall_by_size': dict(sizes),
            'recall_by_class': dict(classes), 'density_bin': density_bin(gt_count), 'image_occlusion': image_occ,
            'gt_above_query_capacity': gt_count > len(scores), 'near_query_capacity': int(keep.sum()) >= len(scores)-100,
            'predictions': {'query_indices': query_indices.tolist(), 'scores': scores[keep].tolist(),
                            'points_xy': points_orig.tolist(), 'boxes_xyxy': boxes_orig.tolist(),
                            'accepted': accepted.tolist(), 'ignored_unmatched': ignored.tolist(),
                            'matched_prediction_indices': matched.tolist(), 'matched_gt_indices': cols.tolist(),
                            'coordinate_space': 'original', 'original_size_hw': [oh, ow]}}


def summarize(entries):
    if not entries:
        return {'images': 0}
    err = np.asarray([x['error'] for x in entries], dtype=float)
    tp, fp, fn = (sum(x[k] for x in entries) for k in ['tp', 'fp', 'fn'])
    result = {'images': len(entries), 'mae': float(np.abs(err).mean()), 'rmse': float(np.sqrt((err**2).mean())),
              'mean_signed_error': float(err.mean()), 'mean_undercount': float(np.maximum(-err, 0).mean()),
              'mean_overcount': float(np.maximum(err, 0).mean()), 'exact_count_rate': float((err == 0).mean()),
              'gt_total': sum(x['gt_count'] for x in entries), 'pred_total': sum(x['pred_count'] for x in entries),
              'tp': tp, 'fp': fp, 'fn': fn, 'precision': tp/(tp+fp) if tp+fp else None,
              'recall': tp/(tp+fn) if tp+fn else None, 'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
              'gt_above_query_capacity_images': sum(x['gt_above_query_capacity'] for x in entries),
              'near_query_capacity_images': sum(x['near_query_capacity'] for x in entries)}
    for name in ['recall_by_occlusion', 'recall_by_size', 'recall_by_class']:
        bins = defaultdict(lambda: [0, 0])
        for row in entries:
            for k, (gt, found) in row[name].items():
                bins[k][0] += gt
                bins[k][1] += found
        result[name] = {k: {'gt': gt, 'matched': found, 'recall': found/gt if gt else None}
                        for k, (gt, found) in bins.items()}
    return result


def aggregate_records(records):
    result = {}
    for mode in ['single', 'pseudo3']:
        modes = {}
        for dataset in sorted({r['dataset'] for r in records}):
            entries = [r['modes'][mode] for r in records if r['dataset'] == dataset]
            item = summarize(entries)
            for group in ['density_bin', 'image_occlusion']:
                item['by_'+group] = {k: summarize([e for e in entries if e[group] == k])
                                    for k in sorted({e[group] for e in entries})}
            modes[dataset] = item
        result[mode] = {'per_dataset': modes,
                        'macro_dataset_mae': float(np.mean([v['mae'] for v in modes.values()])) if modes else None,
                        'note': 'Localization is reported per dataset; point-distance and box-IoU metrics must not be pooled.'}
    return result
