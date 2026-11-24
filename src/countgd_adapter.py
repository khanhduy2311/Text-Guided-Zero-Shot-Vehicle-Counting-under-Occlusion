"""Text-only bridge from UnifiedCountingDataset to CountGD-style forwards.

Inject the backend's native nested_tensor_from_tensor_list and tokenizer. No
global import-path changes, checkpoint loading, or ground-truth model inputs.
"""
import copy
from dataclasses import dataclass, replace
from typing import Any

import torch

from .counting_data.schema import SOURCES, TARGETS


def _move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    return value


def xyxy_to_cxcywh(boxes):
    return torch.cat(((boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]), dim=-1)


@dataclass
class CountGDBatch:
    samples: Any
    captions: list
    cap_list: list
    exemplars: list
    exemplar_labels: list
    targets: list
    token_ids: torch.Tensor
    phrase_mask: torch.Tensor
    purpose: str

    def to(self, device):
        return replace(self, samples=self.samples.to(device),
                       exemplars=_move(self.exemplars, device),
                       exemplar_labels=_move(self.exemplar_labels, device),
                       targets=_move(self.targets, device), token_ids=self.token_ids.to(device),
                       phrase_mask=self.phrase_mask.to(device))

    def native_box_targets(self):
        """Opt-in legacy criterion use only when it can represent the supervision.

        Legacy SetCriterion ignores masks and requires boxes for all labels.
        Refuse point-only/ignored data instead of silently treating it as empty.
        """
        for target in self.targets:
            if target['supervision'] != 'box' or not bool(target['supervision_mask'].all()):
                raise ValueError('Legacy criterion cannot handle point-only or ignore/ROI targets; use CountingCriterion')
        return self.targets


class CountGDAdapter:
    def __init__(self, tokenizer, nested_tensor_factory, *, max_text_len=256,
                 images_are_normalized=True):
        if not getattr(tokenizer, 'is_fast', False):
            raise ValueError('A fast tokenizer is required for exact query/token alignment')
        self.tokenizer = tokenizer
        self.nested_tensor_factory = nested_tensor_factory
        self.max_text_len = max_text_len
        self.images_are_normalized = images_are_normalized

    def prepare(self, images, unified_targets, *, purpose):
        if purpose not in {'train', 'validate', 'evaluate'}:
            raise ValueError('Explicit train/validate/evaluate purpose is required')
        if not images or len(images) != len(unified_targets):
            raise ValueError('Expected a nonempty image/target batch of equal length')
        allowed = {'train': {'source_train'}, 'validate': {'source_val'},
                   'evaluate': {'source_val', 'source_test', 'target_test'}}[purpose]
        captions, categories, targets, prepared_images = [], [], [], []
        for image, original in zip(images, unified_targets):
            if original['role'] not in allowed:
                raise ValueError(f"{purpose} forbids {original['role']}")
            if purpose != 'evaluate' and (original['dataset'] not in SOURCES or original['dataset'] in TARGETS):
                raise ValueError('Protected target or unapproved source dataset')
            if image.ndim != 3 or image.shape[0] != 3 or not image.is_floating_point() or not torch.isfinite(image).all():
                raise ValueError('Images must be finite floating point RGB CHW tensors')
            h, w = image.shape[-2:]
            if original['size_hw'].tolist() != [h, w]:
                raise ValueError('Image and target sizes disagree')
            query = ' '.join(original['prompt'].replace('_', ' ').strip().rstrip('.').split())
            if not query or '.' in query or '?' in query:
                raise ValueError('This adapter accepts one positive query, not a list of sentence-separated prompts')
            captions.append(query + ' .')
            categories.append([query])
            n = int(original['count'])
            points = original['points_normalized']
            if n < 0 or points.shape != (n, 2) or not torch.isfinite(points).all() or not ((points >= 0) & (points <= 1)).all():
                raise ValueError('Invalid point/count supervision')
            if len(original['instance_ids']) != n or len(original['occlusion']) != n or len(original['truncation']) != n:
                raise ValueError('Instance metadata length mismatch')
            if not torch.allclose(original['points_xy'], points * points.new_tensor([w, h]), atol=1e-3, rtol=1e-5):
                raise ValueError('Pixel and normalized point coordinates disagree')
            for name in ('supervision_mask', 'roi_mask'):
                if original[name].shape != (h, w) or original[name].dtype != torch.bool:
                    raise ValueError(f'{name} must be a boolean HW target mask')
            t = copy.deepcopy(original)
            t.update(caption=captions[-1], cap_list=[query], labels=torch.zeros(n, dtype=torch.int64, device=points.device),
                     points=points.clone(), size=original['size_hw'].clone(), orig_size=original['original_size_hw'].clone())
            boxes = original['boxes_normalized_xyxy']
            if original['supervision'] == 'box':
                if boxes.shape != (n, 4) or not torch.equal(original['box_instance_indices'].cpu(), torch.arange(n)):
                    raise ValueError('Box-supervised targets require one aligned box per instance')
                if not torch.isfinite(boxes).all() or not ((boxes >= 0) & (boxes <= 1)).all() or not (boxes[:, 2:] > boxes[:, :2]).all():
                    raise ValueError('Invalid normalized xyxy boxes')
                if not torch.allclose(original['boxes_xyxy'], boxes * boxes.new_tensor([w, h, w, h]), atol=1e-3, rtol=1e-5):
                    raise ValueError('Pixel and normalized boxes disagree')
                t['boxes'] = xyxy_to_cxcywh(boxes)
            elif original['supervision'] == 'point':
                if boxes.numel() or original['box_instance_indices'].numel():
                    raise ValueError('Point-only supervision must not contain manufactured boxes')
                # Deliberately omit "boxes": an unsupported legacy loss must not
                # interpret a positive-count point example as negative detection.
            else:
                raise ValueError('Unknown supervision type')
            # Strip exemplars even if an upstream loader was configured otherwise.
            t['exemplar_boxes_xyxy'] = image.new_empty((0, 4))
            t['exemplars'] = image.new_empty((0, 4))
            targets.append(t)
            if not self.images_are_normalized:
                if not ((image >= 0) & (image <= 1)).all():
                    raise ValueError('Unnormalized images must be in [0,1]')
                image = (image - image.new_tensor([.485, .456, .406])[:, None, None]) / image.new_tensor([.229, .224, .225])[:, None, None]
            prepared_images.append(image)
        encoded = self.tokenizer(captions, padding='longest', return_tensors='pt', return_offsets_mapping=True, truncation=False)
        ids, offsets = encoded['input_ids'], encoded['offset_mapping']
        if ids.shape[1] > self.max_text_len:
            raise ValueError('Caption exceeds model text capacity; refusing silent truncation')
        phrase_mask = torch.zeros((len(images), self.max_text_len), dtype=torch.bool)
        for i, cat in enumerate(categories):
            phrase_mask[i, :ids.shape[1]] = (offsets[i, :, 1] > offsets[i, :, 0]) & (offsets[i, :, 0] < len(cat[0]))
            if not phrase_mask[i].any():
                raise ValueError('Query has no usable text tokens')
        # Factory sees images ONLY. Its mask represents padding, never occlusion.
        samples = self.nested_tensor_factory(prepared_images)
        for i, image in enumerate(prepared_images):
            h, w = image.shape[-2:]
            if bool(samples.mask[i, :h, :w].any()):
                raise ValueError('Backend factory masked real image pixels')
        return CountGDBatch(samples=samples, captions=captions, cap_list=categories,
                            exemplars=[im.new_empty((0, 4)) for im in prepared_images],
                            exemplar_labels=[torch.empty(0, dtype=torch.long, device=im.device) for im in prepared_images],
                            targets=targets, token_ids=ids, phrase_mask=phrase_mask, purpose=purpose)

    def forward(self, model, batch):
        if model.training and batch.purpose != 'train':
            raise ValueError('Call model.eval() before validation/evaluation')
        # Both native CountGD and CountGD++ accept these positional arguments.
        # No targets, GT points/counts, ignore masks or ROI are model arguments.
        outputs = model(batch.samples, batch.exemplars, batch.exemplar_labels, captions=batch.captions)
        aligned_phrase_mask(outputs, batch)
        return outputs

    @torch.no_grad()
    def postprocess(self, outputs, batch, *, score_threshold, token_reduction='max', original_coordinates=True):
        """Decode text-only predictions without GT-dependent filtering or NMS.

        Scores use only the query tokens. Apply benchmark ROI filtering separately
        during evaluation. Returned coordinates are continuous pixel xy/xyxy.
        """
        if not 0 <= score_threshold <= 1 or token_reduction not in {'mean', 'max'}:
            raise ValueError('Invalid score threshold or token reduction')
        mask = aligned_phrase_mask(outputs, batch)
        result = []
        for i, t in enumerate(batch.targets):
            prob = outputs['pred_logits'][i, :, mask[i]].float().sigmoid()
            scores = prob.mean(-1) if token_reduction == 'mean' else prob.max(-1).values
            keep = scores > score_threshold
            b = outputs['pred_boxes'][i, keep].float()
            h, w = t['size'].tolist()
            xyxy = torch.cat((b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2), -1) * b.new_tensor([w, h, w, h])
            points = b[:, :2] * b.new_tensor([w, h])
            if original_coordinates:
                if t['flipped']:
                    x1 = w - xyxy[:, 2].clone(); x2 = w - xyxy[:, 0].clone()
                    xyxy[:, 0], xyxy[:, 2] = x1, x2
                    points[:, 0] = w - points[:, 0]
                sx, sy = t['scale_xy']
                xyxy = xyxy / xyxy.new_tensor([sx, sy, sx, sy])
                points = points / points.new_tensor([sx, sy])
                h, w = t['orig_size'].tolist()
            xyxy[:, 0::2].clamp_(0, w); xyxy[:, 1::2].clamp_(0, h)
            result.append({'id': t['id'], 'prompt': t['prompt'], 'count': int(keep.sum()),
                           'scores': scores[keep], 'points_xy': points, 'boxes_xyxy': xyxy,
                           'query_indices': keep.nonzero(as_tuple=True)[0],
                           'coordinate_space': 'original' if original_coordinates else 'resized'})
        return result


def aligned_phrase_mask(outputs, batch):
    logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
    if logits.ndim != 3 or boxes.shape != (*logits.shape[:2], 4) or logits.shape[0] != len(batch.targets):
        raise ValueError('Unsupported CountGD output shapes')
    if logits.shape[2] > batch.phrase_mask.shape[1]:
        raise ValueError('Output text dimension exceeds adapter capacity')
    if batch.phrase_mask[:, logits.shape[2]:].any():
        raise ValueError('Output truncated query tokens')
    actual = outputs['token']['input_ids']
    if actual.shape != batch.token_ids.shape or not torch.equal(actual.cpu(), batch.token_ids.cpu()):
        raise ValueError('Model and adapter tokenization differ')
    mask = batch.phrase_mask[:, :logits.shape[2]].to(logits.device)
    if 'text_mask' in outputs and not bool(outputs['text_mask'].to(mask.device)[mask].all()):
        raise ValueError('Model masked a query token')
    selected = logits.masked_select(mask[:, None, :].expand_as(logits))
    if not torch.isfinite(selected).all() or not torch.isfinite(boxes).all():
        raise ValueError('Nonfinite predictions on supervised tokens/coordinates')
    if not ((boxes >= 0) & (boxes <= 1)).all():
        raise ValueError('Expected normalized cxcywh predictions')
    return mask
