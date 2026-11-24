"""Paired source-only occlusion evaluation of trained A1/A2 checkpoints."""
import gc
import hashlib
import json
from pathlib import Path

import torch

from .countgdpp_inference import infer_two_pass
from .countgdpp_occlusion import occlude_source
from .counting_eval import evaluate_image, aggregate_records


def evaluate_pair(model, datasets, config, output, root, nested_factory, device, *, write_json, sha):
    output = Path(output); root = Path(root)
    a1_best = json.loads((root/config['reference_a1_best']).read_text())
    a1_path = (root/config['reference_a1_best']).parent/'checkpoint_best.pth'
    a2_best = json.loads((output/'best.json').read_text())
    checkpoints = [('A1', a1_path, a1_best), ('A2', output/'checkpoint_best.pth', a2_best)]
    comparison = {'status': 'running', 'scope': 'fixed source validation vehicle images only',
                  'checkpoint_selection': 'natural source validation single macro MAE, same as A1',
                  'fixed_occlusion_policy': config['occlusion'], 'target_images_loaded': False,
                  'note': 'Synthetic occlusion is generated using source annotations to preserve labels; masks/GT never enter model inference. Bbox visibility is a proxy.',
                  'results': {}}
    write_json(output/'occlusion_comparison.json', comparison)
    model.requires_grad_(False).eval()
    for label, path, best in checkpoints:
        if sha(path) != best['checkpoint_sha256']:
            raise ValueError('Trained checkpoint hash mismatch: ' + label)
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state['format'] != 'countgdpp_a1_v1':
            raise ValueError('Unsupported trained checkpoint format')
        previous = state['identity']['config']
        for key in ('backend', 'bundle', 'sources', 'short_side', 'max_size', 'num_queries', 'amp',
                    'box_threshold', 'text_threshold', 'pseudo_threshold', 'num_pseudo_exemplars'):
            if previous[key] != config[key]: raise ValueError('Unequal A1/A2 inference setting: ' + key)
        model.load_state_dict(state['model'], strict=True)
        del state; gc.collect()
        rows = []; signature = hashlib.sha256(); applied = 0
        destination = output/f'fixed_occlusion_{label}.jsonl'
        with destination.with_suffix('.jsonl.tmp').open('w') as stream:
            for source in ('visdrone', 'ua_detrac'):
                for i in range(len(datasets['val', source])):
                    image, target = datasets['val', source][i]
                    image, _, meta = occlude_source(image, target, config['occlusion'],
                        epoch=0, offset=0, mode='validation')
                    prediction = infer_two_pass(model, image, target['prompt'], nested_factory,
                        device=device, pseudo_threshold=config['pseudo_threshold'], num_exemplars=config['num_pseudo_exemplars'])
                    modes = {}
                    for mode in ('single', 'pseudo3'):
                        result = evaluate_image(prediction[mode], target, threshold=config['box_threshold'],
                            text_threshold=config['text_threshold'], box_iou_threshold=config['box_iou_threshold'],
                            point_radius_fraction=config['point_radius_fraction'])
                        modes[mode] = {k: v for k, v in result.items() if k != 'predictions'}
                    row = {'id': target['id'], 'dataset': target['dataset'], 'synthetic_occlusion': meta, 'modes': modes}
                    rows.append(row); applied += int(meta['applied'])
                    signature.update((target['id']+'\0'+meta.get('mask_sha256', 'unmodified')+'\n').encode())
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                    if len(rows) % 100 == 0:
                        stream.flush()
                        write_json(output/'occlusion_progress.json', {'status': 'running', 'checkpoint': label,
                            'images_completed': len(rows), 'images_expected': 8871})
                        print(f'Fixed occlusion {label}: {len(rows)}/8871', flush=True)
        destination.with_suffix('.jsonl.tmp').replace(destination)
        expected = sum(len(datasets['val', source]) for source in ('visdrone', 'ua_detrac'))
        if len(rows) != expected or len({r['id'] for r in rows}) != expected:
            raise ValueError('Incomplete or duplicate fixed validation coverage')
        metrics = aggregate_records(rows)
        result = {'images': len(rows), 'synthetically_occluded_images': applied,
                  'fixed_images_and_masks_sha256': signature.hexdigest(), 'metrics': metrics,
                  'checkpoint': str(path.relative_to(root)), 'checkpoint_sha256': best['checkpoint_sha256'],
                  'checkpoint_epoch': best['epoch']}
        comparison['results'][label] = result
        write_json(output/f'fixed_occlusion_{label}.metrics.json', result)
        write_json(output/'occlusion_comparison.json', comparison)
    a, b = (comparison['results'][label] for label in ('A1', 'A2'))
    if a['fixed_images_and_masks_sha256'] != b['fixed_images_and_masks_sha256']:
        raise ValueError('Paired evaluation used different images or masks')
    comparison.update(status='completed', paired_masks_verified=True,
        single_mae_delta_A2_minus_A1={source: b['metrics']['single']['per_dataset'][source]['mae']-
            a['metrics']['single']['per_dataset'][source]['mae'] for source in ('VisDrone', 'UA_DETRAC')},
        natural_validation_single_mae_delta_A2_minus_A1={source: a2_best['metrics']['single']['per_dataset'][source]['mae']-
            a1_best['metrics']['single']['per_dataset'][source]['mae'] for source in ('FSC147', 'VisDrone', 'UA_DETRAC')})
    write_json(output/'occlusion_comparison.json', comparison)
    write_json(output/'occlusion_progress.json', {'status': 'completed'})
    return comparison
