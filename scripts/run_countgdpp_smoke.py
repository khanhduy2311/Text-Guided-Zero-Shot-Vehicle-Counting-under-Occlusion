"""Bounded source-only CountGD++ smoke training, with fixed before/after diagnostics.

This is a small engineering run, not a benchmark or final trained model.
Stops on CUDA OOM; resume a saved optimizer boundary into a new output directory.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import inspect
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'CountGDPlusPlus'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import torch
from groundingdino.util.misc import nested_tensor_from_tensor_list
from util.slconfig import SLConfig
from models.GroundingDINO.groundingdino import build_groundingdino
from src.counting_data.loader import UnifiedCountingDataset
from src.countgd_adapter import CountGDAdapter
from src.countgd_criterion import CountingCriterion, CountingLossConfig
from src.countgdpp_checkpoint import load_official_countgdpp

GIB = 2**30


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def choose_records(records, n, seed, config, hard_examples):
    """Include dense/large-area/ignored images, then sample without replacement.

    Counts exceeding the detector capacity are excluded only from this small
    smoke subset, never changed in the frozen manifests or silently truncated.
    """
    eligible = [i for i, r in enumerate(records) if r['count'] <= config['num_queries']]
    if len(eligible) < n:
        raise ValueError('Not enough eligible source records')
    def area(i):
        r = records[i]
        scale = min(config['short_side']/min(r['width'], r['height']), config['max_size']/max(r['width'], r['height']))
        return round(r['width']*scale) * round(r['height']*scale)
    chosen = []
    if hard_examples:
        for key in (lambda i: records[i]['count'], area,
                    lambda i: len(records[i]['ignore_regions'])):
            index = max(eligible, key=key)
            if index not in chosen and len(chosen) < n:
                chosen.append(index)
    rng = random.Random(seed)
    chosen += rng.sample([i for i in eligible if i not in chosen], n-len(chosen))
    eligible_set = set(eligible)
    return chosen, [records[i]['id'] for i in range(len(records)) if i not in eligible_set]


def gradient_audit(model):
    groups = {'input_proj': [], 'text_projection': [], 'encoder': [], 'decoder': [], 'box_head': []}
    missing, nonfinite = [], []
    frozen_with_gradient = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                frozen_with_gradient.append(name)
            continue
        grad = parameter.grad
        if grad is None:
            missing.append(name)
            continue
        if not bool(torch.isfinite(grad).all()):
            nonfinite.append(name)
        norm2 = grad.detach().float().square().sum()
        for group, match in [('input_proj', name.startswith('input_proj.')),
                             ('text_projection', name.startswith('feat_map.')),
                             ('encoder', name.startswith('transformer.encoder.')),
                             ('decoder', name.startswith('transformer.decoder.')),
                             ('box_head', 'bbox_embed' in name)]:
            if match:
                groups[group].append(norm2)
    norms = {k: float(torch.stack(v).sum().sqrt()) if v else 0. for k, v in groups.items()}
    # These branches are unused in text-only mode. All active parameters must
    # receive a gradient; zero gradient values remain possible for some tensors.
    unexpected_missing = [name for name in missing if not name.startswith(('feature_map_', 'label_enc.'))]
    if nonfinite or frozen_with_gradient or unexpected_missing or any(v <= 0 for v in norms.values()):
        raise RuntimeError(f'Gradient audit failed: nonfinite={nonfinite[:5]}, frozen={frozen_with_gradient[:5]}, missing={unexpected_missing[:10]}, norms={norms}')
    return {'group_norms': norms, 'unused_text_only_parameters': missing,
            'all_active_gradients_finite': True, 'frozen_gradients_absent': True}


def run(args, report):
    config = json.loads(args.config.read_text())
    if config.get('backend') != 'CountGD++':
        raise ValueError('This runner requires the CountGD++ backend')
    native_file = Path(inspect.getfile(build_groundingdino)).resolve()
    if not native_file.is_relative_to(ROOT/'CountGDPlusPlus'):
        raise RuntimeError('Wrong backend imported; refusing CountGD fallback')
    report['backend'] = 'CountGD++'
    report['native_model_implementation'] = str(native_file.relative_to(ROOT))
    if config['batch_size'] != 1 or config['amp'] != 'bfloat16':
        raise ValueError('This bounded runner supports batch size 1 and BF16')
    if set(config['sources']) != {'fsc147', 'visdrone', 'ua_detrac'}:
        raise ValueError('Only the three approved source manifests are supported')
    if config['save_every_micro_steps'] % config['gradient_accumulation']:
        raise ValueError('Checkpoints must be at optimizer boundaries')
    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA with BF16 support is required')
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info(device)
    limit = min(config['allocator_limit_gib']*GIB, free-config['free_memory_reserve_gib']*GIB)
    if limit < 3*GIB:
        raise torch.cuda.OutOfMemoryError('Less than 3 GiB available after reserving memory for other processes')
    torch.cuda.set_per_process_memory_fraction(limit/total, device)
    report.update(config=config, initial_free_gib=free/GIB, allocator_limit_gib=limit/GIB,
                  gpu=torch.cuda.get_device_name(device), torch_version=torch.__version__,
                  target_images_loaded=False, micro_steps_completed=0, optimizer_steps_completed=0)
    code_paths = [args.config, ROOT/config['provenance'], ROOT/config['model_config'], Path(__file__),
                  ROOT/'src/countgd_adapter.py', ROOT/'src/countgd_criterion.py', ROOT/'src/countgdpp_checkpoint.py']
    code_paths += list((ROOT/'CountGDPlusPlus/models').rglob('*.py'))
    code_paths += list((ROOT/'CountGDPlusPlus/groundingdino').rglob('*.py'))
    code_paths += list((ROOT/'CountGDPlusPlus/util').rglob('*.py'))
    report['code_sha256'] = {str(p.relative_to(ROOT)): sha256(p) for p in code_paths}
    bundle = ROOT / config['bundle']
    lock = json.loads((bundle/'protocol_lock.json').read_text())
    for name, expected in lock['implementation_sha256'].items():
        if sha256(ROOT/name) != expected:
            raise ValueError(f'Frozen data code changed: {name}')
    report['protocol_lock_sha256'] = sha256(bundle/'protocol_lock.json')
    datasets, selected, selection = {}, {'train': [], 'val': []}, {}
    for split, purpose in [('train', 'train'), ('val', 'validate')]:
        for source_id, source in enumerate(config['sources']):
            name = f'source_{split}_{source}.jsonl'
            if sha256(bundle/name) != lock['artifacts_sha256'][name]:
                raise ValueError(f'Source manifest changed: {name}')
            ds = UnifiedCountingDataset(bundle/name, ROOT, purpose=purpose,
                                        short_side=config['short_side'], max_size=config['max_size'],
                                        horizontal_flip_p=0, text_only=True)
            with (bundle/name).open() as stream:
                records = [json.loads(line) for line in stream if line.strip()]
            amount = config['train_per_source'] if split == 'train' else config['validation_per_source']
            indices, too_dense = choose_records(records, amount, config['seed']+source_id+(100 if split=='val' else 0),
                                                config, hard_examples=split=='train')
            key = split+'_'+source
            datasets[key] = ds
            selected[split] += [(key, i) for i in indices]
            selection[key] = {'manifest': name, 'sha256': lock['artifacts_sha256'][name],
                              'records': [{'index': i, 'id': records[i]['id'], 'count': records[i]['count'],
                                           'prompt': records[i]['prompt']} for i in indices],
                              'excluded_from_smoke_only_above_query_capacity': too_dense}
    write_json(args.output/'sample_selection.json', selection)
    report['selection_sha256'] = sha256(args.output/'sample_selection.json')
    reference = json.loads((ROOT/config['reference_selection']).read_text())
    if reference != selection:
        raise ValueError('CountGD++ smoke subset differs from the previous 48+12 source records')
    report['same_selection_as_countgd_control'] = True
    # Interleave the three sources in each epoch; randomize each source locally.
    order = []
    for epoch in range(config['epochs']):
        rng = random.Random(config['seed']+epoch)
        pools = [[s for s in selected['train'] if s[0] == 'train_'+source] for source in config['sources']]
        for pool in pools:
            rng.shuffle(pool)
        order += [(epoch, sample) for row in zip(*pools) for sample in row]
    if len(order) % config['gradient_accumulation']:
        raise ValueError('Smoke subset must end at an accumulation boundary')
    report.update(train_images=len(selected['train']), validation_images=len(selected['val']), planned_micro_steps=len(order))
    provenance = json.loads((ROOT/config['provenance']).read_text())
    checkpoint_path = ROOT/provenance['path']
    if checkpoint_path.stat().st_size != provenance['size_bytes'] or sha256(checkpoint_path) != provenance['sha256']:
        raise ValueError('Initialization checkpoint does not match the verified upstream artifact')
    report['pretrained'] = provenance
    revision = subprocess.check_output(['git', '-C', str(ROOT/'CountGDPlusPlus'), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != provenance['code_revision']:
        raise ValueError('CountGD++ repository revision differs from pinned provenance')
    report['native_code_revision'] = revision
    patch = subprocess.check_output(['git', '-C', str(ROOT/'CountGDPlusPlus'), 'diff', '--', '*.py'], text=True)
    (args.output/'runtime_patch.diff').write_text(patch)
    cfg = SLConfig.fromfile(str(ROOT/config['model_config']))
    cfg.device = str(device)
    cfg.text_encoder_type = str(ROOT/'CountGD/checkpoints/bert-base-uncased')
    cfg.use_checkpoint = config['gradient_checkpointing']
    cfg.use_transformer_ckpt = config['gradient_checkpointing']
    if cfg.num_queries != config['num_queries']:
        raise ValueError('Query capacity differs from sample-selection policy')
    print('Building native CountGD++ and loading verified official CountGD++ weights', flush=True)
    model, _, _ = build_groundingdino(cfg)
    report['checkpoint_loading'] = load_official_countgdpp(model, checkpoint_path)
    for name, parameter in model.named_parameters():
        if any(key in name for key in config['freeze_keywords']):
            parameter.requires_grad_(False)
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    report['trainable_parameters'] = sum(p.numel() for p in trainable)
    report['total_parameters'] = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=config['learning_rate'], weight_decay=config['weight_decay'], foreach=False)
    adapter = CountGDAdapter(model.tokenizer, nested_tensor_from_tensor_list)
    criterion = CountingCriterion(CountingLossConfig(**config['loss_config'])).to(device)
    start_micro = 0
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=True)
        if (saved['config'] != config or saved['selection_sha256'] != report['selection_sha256']
                or saved['pretrained_sha256'] != provenance['sha256']
                or saved['code_sha256'] != report['code_sha256']):
            raise ValueError('Resume code, configuration, sample selection or initialization changed')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        start_micro = saved['next_micro_step']
        torch.set_rng_state(saved['rng_cpu'])
        torch.cuda.set_rng_state(saved['rng_cuda'], device)
        report.update(resumed_from=str(args.resume), micro_steps_completed=start_micro,
                      optimizer_steps_completed=start_micro//config['gradient_accumulation'], before=saved['before'])
        del saved

    def make_batch(sample):
        key, index = sample
        image, target = datasets[key][index]
        purpose = 'train' if key.startswith('train_') else 'validate'
        return adapter.prepare([image], [target], purpose=purpose).to(device)

    def evaluate(samples, tag):
        model.eval(); criterion.eval()
        rows = []
        with torch.inference_mode():
            for sample in samples:
                batch = make_batch(sample)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    outputs = adapter.forward(model, batch)
                losses = criterion(outputs, batch)
                rows.append({'id': batch.targets[0]['id'], 'dataset': batch.targets[0]['dataset'],
                             'losses': {k: float(v) for k, v in losses.items()}})
        result = {'images': len(rows), 'mean_loss': sum(r['losses']['loss_total'] for r in rows)/len(rows),
                  'per_dataset': {ds: sum(r['losses']['loss_total'] for r in rows if r['dataset']==ds)/sum(r['dataset']==ds for r in rows)
                                  for ds in sorted({r['dataset'] for r in rows})}, 'records': rows}
        write_json(args.output/(tag+'.json'), result)
        print(f'{tag}: {len(rows)} images, mean loss={result["mean_loss"]:.6f}', flush=True)
        return {k:v for k,v in result.items() if k != 'records'}

    def save_checkpoint(next_micro):
        value = {'model': cpu_tree(model.state_dict()), 'optimizer': cpu_tree(optimizer.state_dict()),
                 'next_micro_step': next_micro, 'config': config, 'selection_sha256': report['selection_sha256'],
                 'pretrained_sha256': provenance['sha256'], 'code_sha256': report['code_sha256'], 'rng_cpu': torch.get_rng_state(),
                 'rng_cuda': torch.cuda.get_rng_state(device), 'before': report['before'],
                 'purpose': 'source smoke test; not a benchmark-selected model'}
        path = args.output/'checkpoint_last.pth'
        temporary = path.with_suffix('.pth.tmp')
        torch.save(value, temporary)
        temporary.replace(path)
        report['last_checkpoint'] = str(path.relative_to(ROOT))
        report['last_checkpoint_micro_step'] = next_micro
        write_json(args.output/'report.json', report)
        print(f'Saved checkpoint after {next_micro} micro steps', flush=True)

    if not args.resume:
        report['before'] = {'train': evaluate(selected['train'], 'before_train'),
                            'val': evaluate(selected['val'], 'before_val')}
    save_checkpoint(start_micro)
    model.train(); model.backbone.eval(); model.bert.eval(); criterion.train()
    optimizer.zero_grad(set_to_none=True)
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    for micro in range(start_micro, len(order)):
        epoch, sample = order[micro]
        started = time.perf_counter()
        batch = make_batch(sample)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = adapter.forward(model, batch)
        losses = criterion(outputs, batch)
        (losses['loss_total']/config['gradient_accumulation']).backward()
        row = {'micro_step': micro+1, 'epoch': epoch+1, 'id': batch.targets[0]['id'],
               'dataset': batch.targets[0]['dataset'], 'count': batch.targets[0]['count'],
               'size_hw': batch.targets[0]['size'].tolist(),
               'losses': {k:float(v) for k,v in losses.items()}}
        if (micro+1) % config['gradient_accumulation'] == 0:
            audit = gradient_audit(model)
            row['gradient_audit'] = audit
            row['gradient_norm_before_clip'] = float(torch.nn.utils.clip_grad_norm_(trainable, config['gradient_clip'], error_if_nonfinite=True))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            report['optimizer_steps_completed'] += 1
            if 'first_gradient_audit' not in report:
                report['first_gradient_audit'] = audit
            report['last_gradient_audit'] = audit
        torch.cuda.synchronize(device)
        row.update(seconds=time.perf_counter()-started,
                   peak_allocated_gib=torch.cuda.max_memory_allocated(device)/GIB,
                   peak_reserved_gib=torch.cuda.max_memory_reserved(device)/GIB)
        report.update(micro_steps_completed=micro+1, peak_allocated_gib=row['peak_allocated_gib'],
                      peak_reserved_gib=row['peak_reserved_gib'])
        with (args.output/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        del batch, outputs, losses
        if (micro+1) % config['gradient_accumulation'] == 0:
            print(f'Step {micro+1}/{len(order)}; update {report["optimizer_steps_completed"]}; loss={row["losses"]["loss_total"]:.4f}; peak={row["peak_reserved_gib"]:.2f} GiB', flush=True)
        if (micro+1) % config['save_every_micro_steps'] == 0 or micro+1 == len(order):
            save_checkpoint(micro+1)
    report['after'] = {'train': evaluate(selected['train'], 'after_train'),
                       'val': evaluate(selected['val'], 'after_val')}
    before, after = report['before']['train']['mean_loss'], report['after']['train']['mean_loss']
    report['fixed_train_loss_reduction_percent'] = 100*(before-after)/max(abs(before), 1e-12)
    report['engineering_checks_passed'] = report['optimizer_steps_completed'] == len(order)//config['gradient_accumulation']
    report['train_loss_decreased'] = after < before
    report['status'] = 'passed' if report['engineering_checks_passed'] and after < before else 'completed_requires_review'
    print(f'Completed: {report["status"]}; fixed train loss {before:.6f} -> {after:.6f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/countgdpp_smoke_v1.json')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/countgdpp_smoke_v1')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.config = args.config.resolve(); args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError('Use a new output directory; existing experiment results are preserved')
    args.output.mkdir(parents=True)
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
              'scope': 'CountGD++ source-only engineering run, not target evaluation or final baseline'}
    start = time.perf_counter()
    try:
        run(args, report)
    except torch.cuda.OutOfMemoryError as exc:
        report.update(status='stopped_oom', error=str(exc),
                      restart=('Resume the saved optimizer boundary into a new output directory when memory is available.'
                               if report.get('last_checkpoint') else 'Rerun from initialization when memory is available; no checkpoint was saved.'))
        print('CUDA memory limit reached. Stopped without an automatic retry.', flush=True)
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter()-start
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(args.output/'report.json', report)
        print(f'Report: {args.output / "report.json"}', flush=True)


if __name__ == '__main__':
    main()
