"""Source-only loss bridge for mixed box/point CountGD supervision.

This is an explicit experimental criterion, not the unmodified CountGD loss.
Only unmatched negative queries whose centers are outside supervision_mask are
ignored. Annotated positives always take precedence. Masks never enter the model.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from .countgd_adapter import aligned_phrase_mask
from .counting_data.schema import SOURCES


@dataclass(frozen=True)
class CountingLossConfig:
    cost_class: float = 5.
    cost_geometry: float = 1.
    cost_giou: float = 0.
    weight_class: float = 5.
    weight_box: float = 1.
    weight_point: float = 1.
    weight_giou: float = 0.
    alpha: float = .25
    gamma: float = 2.
    auxiliary_weight: float = 1.
    intermediate_weight: float = 1.


def generalized_box_iou(boxes1, boxes2):
    def corners(b):
        return torch.cat((b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2), -1)
    a, b = corners(boxes1), corners(boxes2)
    lt, rb = torch.maximum(a[:, None, :2], b[None, :, :2]), torch.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = (rb - lt).clamp(min=0).prod(-1)
    union = boxes1[:, None, 2:].prod(-1) + boxes2[None, :, 2:].prod(-1) - intersection
    enclosure = (torch.maximum(a[:, None, 2:], b[None, :, 2:]) - torch.minimum(a[:, None, :2], b[None, :, :2])).clamp(min=0).prod(-1)
    return intersection / union.clamp(min=1e-7) - (enclosure - union) / enclosure.clamp(min=1e-7)


def _normalizer(count, device):
    value = torch.tensor(float(count), device=device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(value)
        value /= torch.distributed.get_world_size()
    return value.clamp(min=1.)


class CountingCriterion(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or CountingLossConfig()
        c = self.config
        if not 0 < c.alpha < 1 or c.gamma < 0:
            raise ValueError('Invalid focal loss parameters')
        if min(c.cost_class, c.cost_geometry, c.cost_giou, c.weight_class,
               c.weight_box, c.weight_point, c.weight_giou,
               c.auxiliary_weight, c.intermediate_weight) < 0:
            raise ValueError('Loss weights/costs must be nonnegative')
        if c.cost_class + c.cost_geometry + c.cost_giou == 0:
            raise ValueError('At least one matching cost must be positive')

    def _layer(self, outputs, batch):
        mask = aligned_phrase_mask(outputs, batch)
        boxes_all = outputs['pred_boxes'].float()
        # Never multiply padded -inf logits by zero: that would yield NaN.
        zero = boxes_all.sum() * 0 + outputs['pred_logits'].masked_select(mask[:, None, :].expand_as(outputs['pred_logits'])).float().sum() * 0
        losses = {name: zero for name in ('loss_class', 'loss_box', 'loss_point', 'loss_giou')}
        c = self.config
        for i, target in enumerate(batch.targets):
            logits = outputs['pred_logits'][i, :, mask[i]].float()
            boxes = boxes_all[i]
            n = target['count']
            if n > len(boxes):
                raise ValueError(f'{n} instances exceed {len(boxes)} queries; use a declared crop policy, never truncate GT')
            with torch.no_grad():
                if n:
                    p = logits.sigmoid()
                    negative = (1-c.alpha) * p.pow(c.gamma) * F.softplus(logits)
                    positive = c.alpha * (1-p).pow(c.gamma) * F.softplus(-logits)
                    cost = c.cost_class * (positive - negative).mean(-1, keepdim=True)
                    if target['supervision'] == 'box':
                        gt = target['boxes'].float()
                        cost = cost + c.cost_geometry * torch.cdist(boxes, gt, p=1)
                        if c.cost_giou:
                            cost = cost - c.cost_giou * generalized_box_iou(boxes, gt)
                    else:
                        cost = cost + c.cost_geometry * torch.cdist(boxes[:, :2], target['points'].float(), p=1)
                    if not torch.isfinite(cost).all():
                        raise ValueError('Nonfinite matching cost')
                    row, col = linear_sum_assignment(cost.cpu().numpy())
                    row = torch.as_tensor(row, dtype=torch.long, device=boxes.device)
                    col = torch.as_tensor(col, dtype=torch.long, device=boxes.device)
                else:
                    row = col = torch.empty(0, dtype=torch.long, device=boxes.device)
                h, w = target['size'].tolist()
                x = (boxes[:, 0] * w).floor().long().clamp(0, w-1)
                y = (boxes[:, 1] * h).floor().long().clamp(0, h-1)
                valid = target['supervision_mask'][y, x].clone()
                valid[row] = True
                expected = torch.zeros_like(logits)
                expected[row] = 1
            p = logits.sigmoid()
            pt = p * expected + (1-p) * (1-expected)
            alpha = c.alpha * expected + (1-c.alpha) * (1-expected)
            focal = F.binary_cross_entropy_with_logits(logits, expected, reduction='none') * (1-pt).pow(c.gamma) * alpha
            # Mean over phrase tokens keeps the scale independent of wordpieces.
            losses['loss_class'] = losses['loss_class'] + focal.mean(-1)[valid].sum()
            if n and target['supervision'] == 'box':
                gt = target['boxes'][col].float()
                losses['loss_box'] = losses['loss_box'] + F.l1_loss(boxes[row], gt, reduction='sum')
                if c.weight_giou:
                    losses['loss_giou'] = losses['loss_giou'] + (1-generalized_box_iou(boxes[row], gt).diag()).sum()
            elif n:
                losses['loss_point'] = losses['loss_point'] + F.l1_loss(boxes[row, :2], target['points'][col].float(), reduction='sum')
        total = sum(t['count'] for t in batch.targets)
        boxed = sum(t['count'] for t in batch.targets if t['supervision'] == 'box')
        device = boxes_all.device
        losses['loss_class'] = losses['loss_class'] / _normalizer(total, device)
        losses['loss_box'] = losses['loss_box'] / _normalizer(boxed, device)
        losses['loss_giou'] = losses['loss_giou'] / _normalizer(boxed, device)
        losses['loss_point'] = losses['loss_point'] / _normalizer(total-boxed, device)
        return losses

    def forward(self, outputs, batch):
        allowed = {'source_train'} if self.training else {'source_train', 'source_val'}
        if batch.purpose == 'evaluate' or any(t['dataset'] not in SOURCES or t['role'] not in allowed for t in batch.targets):
            raise ValueError('Loss is restricted to source train/validation; target/test fitting is forbidden')
        if outputs.get('dn_meta') is not None or 'dn_outputs' in outputs:
            raise ValueError('Denoising supervision is not supported by this criterion')
        c = self.config
        weights = {'loss_class': c.weight_class, 'loss_box': c.weight_box,
                   'loss_point': c.weight_point, 'loss_giou': c.weight_giou}
        losses = self._layer(outputs, batch)
        total = sum(weights[k] * v for k, v in losses.items())
        layers = [(f'_aux_{i}', layer, c.auxiliary_weight) for i, layer in enumerate(outputs.get('aux_outputs', []))]
        if 'interm_outputs' in outputs:
            layers.append(('_interm', outputs['interm_outputs'], c.intermediate_weight))
        for suffix, layer, weight in layers:
            if not weight:
                continue
            layer = dict(layer, token=outputs['token'], text_mask=outputs.get('text_mask', batch.phrase_mask))
            values = self._layer(layer, batch)
            total = total + weight * sum(weights[k] * v for k, v in values.items())
            losses.update({k+suffix: v for k, v in values.items()})
        if not torch.isfinite(total):
            raise ValueError('Nonfinite counting loss')
        losses['loss_total'] = total
        return losses
