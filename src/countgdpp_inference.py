"""Annotation-free CountGD++ inference with the official top-k pseudo-exemplar rule.

Scoring follows engine_inference.py: max over output tokens, then the positive
phrase token threshold. Visual '*' tokens participate in the second-pass score.
No annotations, ROI, or ignore regions are accepted by this module.
"""
import time
import torch


def select_pseudo_exemplars(boxes, scores, size_hw, threshold=0.23, count=3):
    """Official get_pseudo_exemplars selection, for a single image, on any device."""
    indices = (scores > threshold).nonzero(as_tuple=True)[0]
    indices = indices[torch.sort(scores[indices], descending=True).indices][:count]
    chosen = boxes[indices]
    h, w = size_hw
    center = chosen[:, :2] * chosen.new_tensor([w, h])
    size = chosen[:, 2:] * chosen.new_tensor([w, h])
    xyxy = torch.cat((center-size/2, center+size/2), -1)
    xyxy[:, 0::2].clamp_(0, w)
    xyxy[:, 1::2].clamp_(0, h)
    return xyxy, indices


def decode_outputs(outputs, expected_ids, num_exemplars):
    ids = outputs['token']['input_ids'][0].tolist()
    expected = list(expected_ids)
    sep = expected.index(1012)  # BERT '.' token; insertion point in native code.
    expected[sep:sep] = [1008] * num_exemplars
    if ids != expected:
        raise ValueError('Native positive/visual token alignment differs from expected caption')
    logits = outputs['pred_logits'][0].float()
    boxes = outputs['pred_boxes'][0].float()
    if logits.ndim != 2 or boxes.shape != (len(logits), 4):
        raise ValueError('Unexpected prediction shape')
    if torch.isnan(logits).any() or torch.isposinf(logits).any() or not torch.isfinite(boxes).all():
        raise ValueError('Nonfinite model outputs')
    if not ((boxes >= 0) & (boxes <= 1)).all():
        raise ValueError('Expected normalized cxcywh')
    probs = logits.sigmoid()
    # This includes inserted visual tokens, as in the official evaluator.
    text_min = probs[:, 1:ids.index(1012)].min(-1).values
    return boxes, probs.max(-1).values, text_min


@torch.inference_mode()
def infer_two_pass(model, image, prompt, nested_tensor_factory, *, device,
                   pseudo_threshold=0.23, num_exemplars=3):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Evaluation model must be eval() with all parameters frozen')
    query = ' '.join(prompt.replace('_', ' ').strip().rstrip('.').split())
    if not query or '.' in query or '?' in query:
        raise ValueError('Expected one positive query')
    caption = query + ' .'
    ids = model.tokenizer(caption)['input_ids']
    if len(ids) + num_exemplars > model.max_text_len:
        raise ValueError('Prompt would be truncated after adding visual tokens')
    samples = nested_tensor_factory([image.to(device)])
    empty = image.new_empty((0, 4), device=device)
    labels = torch.empty(0, dtype=torch.long, device=device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        outputs = model(samples, [empty], [labels], captions=[caption])
    torch.cuda.synchronize(device)
    first_seconds = time.perf_counter()-start
    boxes, scores, text_min = decode_outputs(outputs, ids, 0)
    pseudo, indices = select_pseudo_exemplars(boxes, scores, image.shape[-2:], pseudo_threshold, num_exemplars)
    first = {k: v.detach().cpu().numpy() for k, v in
             [('boxes', boxes), ('scores', scores), ('text_min', text_min)]}
    del outputs, boxes, scores, text_min
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        outputs = model(samples, [pseudo], [torch.zeros(len(pseudo), dtype=torch.long, device=device)],
                        captions=[caption])
    torch.cuda.synchronize(device)
    second_seconds = time.perf_counter()-start
    boxes, scores, text_min = decode_outputs(outputs, ids, len(pseudo))
    second = {k: v.detach().cpu().numpy() for k, v in
              [('boxes', boxes), ('scores', scores), ('text_min', text_min)]}
    result = {'single': first, 'pseudo3': second,
              'pseudo_boxes_resized': pseudo.detach().cpu().numpy(),
              'pseudo_query_indices': indices.detach().cpu().numpy(),
              'first_forward_seconds': first_seconds, 'second_forward_seconds': second_seconds}
    del outputs, samples, boxes, scores, text_min, pseudo
    return result
