"""A1 source-only training. Profiles before training; stops without retry on OOM.

One balanced epoch covers all eligible source images with 50/25/25 sampling.
Resume only at saved optimizer boundaries into a new output directory.
"""
import argparse
import fcntl
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'CountGDPlusPlus'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import numpy as np
import torch
from groundingdino.util.misc import nested_tensor_from_tensor_list
from util.slconfig import SLConfig
from models.GroundingDINO.groundingdino import build_groundingdino
from models.GroundingDINO import ms_deform_attn
from src.counting_data.loader import UnifiedCountingDataset
from src.countgd_adapter import CountGDAdapter
from src.countgd_criterion import CountingCriterion, CountingLossConfig
from src.countgdpp_checkpoint import load_official_countgdpp
from src.countgdpp_inference import infer_two_pass
from src.counting_eval import evaluate_image, aggregate_records
from src.countgdpp_training import (training_pools, epoch_schedule, cpu_tree, capture_rng,
    restore_rng, save_training_checkpoint, load_training_checkpoint, learning_rate_factor, trees_equal)
from scripts.run_countgdpp_smoke import gradient_audit

GIB = 2**30


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def gpu_lock(args, report):
    uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not uuid.startswith('GPU-') or ',' in uuid:
        raise ValueError('CUDA_VISIBLE_DEVICES must select exactly one GPU UUID')
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                   '--format=csv,noheader,nounits'], text=True).splitlines()
    row = next(r.split(', ') for r in rows if uuid in r)
    if int(row[0]) != args.gpu_index:
        raise ValueError('Host GPU index does not match UUID')
    folder = Path('/tmp') / f'gpu-dispatch-{os.getuid()}'
    folder.mkdir(mode=0o700, exist_ok=True)
    handle = (folder / (uuid + '.lock')).open('a')
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    report.update(hostname=socket.gethostname(), pid=os.getpid(), physical_gpu_index=args.gpu_index,
                  gpu_uuid=uuid, python_executable=sys.executable)
    return handle


def run(args, report):
    config = json.loads(args.config.read_text())
    if (config['backend'], config['amp'], config['batch_size'], config['gradient_accumulation']) != ('CountGD++', 'bfloat16', 1, 8):
        raise ValueError('A1 v1 requires CountGD++, BF16, batch 1, accumulation 8')
    if set(config['sources']) != {'fsc147', 'visdrone', 'ua_detrac'} or config['epochs'] < 1:
        raise ValueError('Invalid source selection or epoch budget')
    torch.set_num_threads(4)
    torch.manual_seed(config['seed']); random.seed(config['seed']); np.random.seed(config['seed'])
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device)[0] < 8 or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A1 v1 requires a GPU with native BF16 support')
    if ms_deform_attn._C is None:
        raise RuntimeError('Expected the verified native CUDA attention extension')
    native = Path(inspect.getfile(build_groundingdino)).resolve()
    if not native.is_relative_to(ROOT / 'CountGDPlusPlus'):
        raise RuntimeError('Wrong model implementation imported')
    samples = []
    for i in range(3):
        samples.append(torch.cuda.mem_get_info(device)[0] / GIB)
        if i < 2:
            time.sleep(2)
    budget = config['allocator_limit_gib']
    reserve = max(config['free_memory_reserve_gib'], budget * .25)
    if min(samples) < budget + reserve:
        raise RuntimeError(f'Insufficient free VRAM: {samples}, requires {budget + reserve} GiB')
    _, total = torch.cuda.mem_get_info(device)
    torch.cuda.set_per_process_memory_fraction(budget * GIB / total, device)
    report.update(config=config, status='preflight', free_gib_samples=samples,
                  memory_budget_gib=budget, memory_reserve_gib=reserve,
                  gpu=torch.cuda.get_device_name(device), target_images_loaded=False,
                  training_initializer='official CountGD++ before local smoke', started_utc=utc())
    write_json(args.output/'report.json', report)
    bundle = ROOT/config['bundle']
    lock = json.loads((bundle/'protocol_lock.json').read_text())
    for name, expected in lock['implementation_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Frozen data code changed: ' + name)
    datasets, records, manifest_hashes = {}, {}, {}
    for split, purpose in [('train', 'train'), ('val', 'validate')]:
        for source in config['sources']:
            name = f'source_{split}_{source}.jsonl'
            digest = sha(bundle/name)
            if digest != lock['artifacts_sha256'][name]:
                raise ValueError('Frozen manifest changed: ' + name)
            manifest_hashes[name] = digest
            datasets[split, source] = UnifiedCountingDataset(bundle/name, ROOT, purpose=purpose,
                short_side=config['short_side'], max_size=config['max_size'], text_only=True,
                horizontal_flip_p=config['horizontal_flip_p'] if split == 'train' else 0.)
            if split == 'train':
                records[source] = [json.loads(line) for line in (bundle/name).open() if line.strip()]
    pools, exclusions = training_pools(records, config['num_queries'])
    orders = [epoch_schedule(pools, config['seed'], epoch) for epoch in range(config['epochs'])]
    total_updates = sum(map(len, orders)) // config['gradient_accumulation']
    selection = {'eligible_per_source': {s: len(v) for s, v in pools.items()},
                 'exclusions': exclusions, 'policy': config['dense_training_policy'],
                 'samples_per_balanced_epoch': len(orders[0]), 'planned_optimizer_updates': total_updates,
                 'validation_per_source': {s: len(datasets['val', s]) for s in config['sources']},
                 'epoch_schedule_sha256': [hashlib.sha256(json.dumps(o).encode()).hexdigest() for o in orders]}
    write_json(args.output/'selection.json', selection)
    provenance = json.loads((ROOT/config['provenance']).read_text())
    checkpoint = ROOT/provenance['path']
    if checkpoint.stat().st_size != provenance['size_bytes'] or sha(checkpoint) != provenance['sha256']:
        raise ValueError('Official initialization checksum mismatch')
    # This known project repository is shared between different NAS/SSH users.
    # Scope Git's ownership exception to this invocation and this exact path.
    git = ['git', '-c', 'safe.directory=' + str(ROOT/'CountGDPlusPlus'), '-C', str(ROOT/'CountGDPlusPlus')]
    revision = subprocess.check_output([*git, 'rev-parse', 'HEAD'], text=True).strip()
    if revision != provenance['code_revision']:
        raise ValueError('CountGD++ revision changed')
    code = [Path(__file__), args.config, ROOT/config['model_config'], ROOT/config['provenance'],
            ROOT/'scripts/run_countgdpp_smoke.py', *sorted((ROOT/'src').glob('count*.py')),
            *sorted((ROOT/'src/counting_data').glob('*.py'))]
    for directory in ('models', 'groundingdino', 'util'):
        code.extend(sorted((ROOT/'CountGDPlusPlus'/directory).rglob('*.py')))
    extension_path = Path(ms_deform_attn._C.__file__)
    identity = {'config': config, 'code_sha256': {str(p.relative_to(ROOT)): sha(p) for p in code},
                'manifest_sha256': manifest_hashes, 'protocol_sha256': sha(bundle/'protocol_lock.json'),
                'selection': selection, 'pretrained_sha256': provenance['sha256'],
                'native_revision': revision, 'torch': str(torch.__version__),
                'cuda_extension_sha256': sha(extension_path)}
    write_json(args.output/'identity.json', identity)
    for p in code:
        dest = args.output/'source_snapshot'/p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dest)
    (args.output/'runtime_patch.diff').write_text(subprocess.check_output(
        [*git, 'diff', '--', '*.py'], text=True))
    cfg = SLConfig.fromfile(str(ROOT/config['model_config']))
    cfg.device = str(device); cfg.text_encoder_type = str(ROOT/'CountGD/checkpoints/bert-base-uncased')
    cfg.use_checkpoint = cfg.use_transformer_ckpt = config['gradient_checkpointing']
    if cfg.num_queries != config['num_queries']:
        raise ValueError('Query capacity mismatch')
    print('Building official CountGD++ for A1', flush=True)
    model, _, _ = build_groundingdino(cfg)
    report['checkpoint_loading'] = load_official_countgdpp(model, checkpoint)
    for name, parameter in model.named_parameters():
        if any(key in name for key in config['freeze_keywords']):
            parameter.requires_grad_(False)
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    adapter = CountGDAdapter(model.tokenizer, nested_tensor_from_tensor_list)
    criterion = CountingCriterion(CountingLossConfig(**config['loss_config'])).to(device)

    def make_optimizer():
        opt = torch.optim.AdamW(trainable, lr=config['learning_rate'], weight_decay=config['weight_decay'], foreach=False)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: learning_rate_factor(step, config['warmup_updates'], total_updates))
        return opt, sched

    optimizer, scheduler = make_optimizer()

    def train_mode():
        model.train(); model.backbone.eval(); model.bert.eval(); criterion.train()

    def batch_for(source, index, epoch, offset):
        # Geometry is a pure function of schedule position, independent of resume or validation.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config['seed'] + epoch*1000003 + offset)
            image, target = datasets['train', source][index]
        return adapter.prepare([image], [target], purpose='train').to(device)

    def backward_sample(source, index, epoch, offset, divisor):
        batch = batch_for(source, index, epoch, offset)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = adapter.forward(model, batch)
        losses = criterion(outputs, batch)
        (losses['loss_total']/divisor).backward()
        return {'id': batch.targets[0]['id'], 'source': source, 'count': batch.targets[0]['count'],
                'loss': float(losses['loss_total'].detach())}

    def validation(epoch, profile=False):
        flags = [p.requires_grad for p in model.parameters()]
        rng = capture_rng(); model.eval(); model.requires_grad_(False)
        rows = []
        path = args.output/(f'validation_epoch_{epoch:03d}.jsonl' if not profile else 'profile_validation.jsonl')
        try:
            with path.with_suffix('.jsonl.tmp').open('w') as stream:
                for source in config['sources']:
                    ds = datasets['val', source]
                    indices = range(len(ds))
                    if profile:
                        # Largest GT image exercises >900 ground truth evaluation without truncation.
                        meta = [ds.record(i) for i in range(len(ds))]
                        indices = sorted({0, max(range(len(meta)), key=lambda i: meta[i]['count'])})
                    for i in indices:
                        image, target = ds[i]
                        prediction = infer_two_pass(model, image, target['prompt'], nested_tensor_from_tensor_list,
                            device=device, pseudo_threshold=config['pseudo_threshold'], num_exemplars=config['num_pseudo_exemplars'])
                        modes = {}
                        for mode in ('single', 'pseudo3'):
                            result = evaluate_image(prediction[mode], target, threshold=config['box_threshold'],
                                text_threshold=config['text_threshold'], box_iou_threshold=config['box_iou_threshold'],
                                point_radius_fraction=config['point_radius_fraction'])
                            modes[mode] = {k: v for k, v in result.items() if k != 'predictions'}
                        row = {'id': target['id'], 'dataset': target['dataset'], 'modes': modes}
                        stream.write(json.dumps(row, allow_nan=False)+'\n'); rows.append(row)
                        if len(rows) % 100 == 0:
                            stream.flush(); report['validation_images_completed'] = len(rows)
                            report['updated_utc'] = utc(); write_json(args.output/'report.json', report)
                            print(f'Validation epoch={epoch}, images={len(rows)}', flush=True)
            path.with_suffix('.jsonl.tmp').replace(path)
            metrics = aggregate_records(rows)
            write_json(path.with_suffix('.metrics.json'), {'images': len(rows), 'metrics': metrics, 'profile_only': profile})
            return metrics
        finally:
            for p, flag in zip(model.parameters(), flags):
                p.requires_grad_(flag)
            restore_rng(rng); train_mode()

    report.update(planned_epochs=config['epochs'], samples_per_epoch=len(orders[0]),
                  planned_optimizer_updates=total_updates, micro_steps_completed=0, optimizer_steps_completed=0,
                  trainable_parameters=sum(p.numel() for p in trainable), status='profiling')
    write_json(args.output/'report.json', report)
    progress = {'epoch': 0, 'next_micro': 0, 'global_update': 0, 'phase': 'train', 'best_score': None}
    if args.resume:
        progress = load_training_checkpoint(args.resume, model, optimizer, scheduler, identity=identity)
        report['resumed_from'] = str(args.resume)
        report.update(optimizer_steps_completed=progress['global_update'],
                      micro_steps_completed=progress['global_update'] * config['gradient_accumulation'])
    else:
        train_mode(); optimizer.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats(device)
        hard = []
        for source in config['sources']:
            def area(i):
                r = records[source][i]
                scale = min(config['short_side']/min(r['width'], r['height']), config['max_size']/max(r['width'], r['height']))
                return round(r['width']*scale)*round(r['height']*scale)
            for key in (lambda i: records[source][i]['count'], area, lambda i: len(records[source][i]['ignore_regions'])):
                sample = source, max(pools[source], key=key)
                if sample not in hard: hard.append(sample)
            empty = next((i for i in pools[source] if records[source][i]['count'] == 0), None)
            if empty is not None: hard.append((source, empty))
        profile = {'train_samples': []}
        for offset, (source, index) in enumerate(hard):
            profile['train_samples'].append(backward_sample(source, index, 0, offset, len(hard)))
            print('Profile backward:', profile['train_samples'][-1], flush=True)
        profile['gradient_audit'] = gradient_audit(model)
        torch.nn.utils.clip_grad_norm_(trainable, config['gradient_clip'], error_if_nonfinite=True)
        optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        probe = args.output/'profile_resume.pth'
        save_training_checkpoint(probe, model, optimizer, scheduler, identity=identity, progress=progress)
        source, index = hard[0]
        rng = capture_rng()
        first = backward_sample(source, index, 0, 10000, 1)['loss']
        grad_first = next(p for p in trainable if p.grad is not None).grad.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        restore_rng(rng)
        control = backward_sample(source, index, 0, 10000, 1)['loss']
        grad_control = next(p for p in trainable if p.grad is not None).grad.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        load_training_checkpoint(probe, model, optimizer, scheduler, identity=identity)
        saved = torch.load(probe, map_location='cpu', weights_only=True)
        for key, actual in [('model', model.state_dict()), ('optimizer', optimizer.state_dict()),
                            ('scheduler', scheduler.state_dict()), ('rng', capture_rng())]:
            if not trees_equal(actual, saved[key]):
                raise RuntimeError('Checkpoint did not restore exact ' + key + ' state')
        del saved
        second = backward_sample(source, index, 0, 10000, 1)['loss']
        grad_second = next(p for p in trainable if p.grad is not None).grad
        comparisons = {}
        for name, gradient in [('repeat_without_load', grad_control), ('repeat_after_load', grad_second)]:
            x, y = grad_first.float().flatten(), gradient.float().flatten()
            relative = float((x-y).norm()/x.norm().clamp_min(1e-12))
            cosine = float(torch.nn.functional.cosine_similarity(x, y, dim=0))
            comparisons[name] = {'relative_l2': relative, 'cosine': cosine,
                                 'max_abs': float((x-y).abs().max())}
            # A no-load repeat control measured ~0.1% L2 variability in native BF16
            # backward. Audit state exactly; allow at most 0.5% gradient L2 drift.
            if not math.isfinite(relative) or relative > .005 or cosine < .99999:
                raise RuntimeError('GPU resume/repeat gradient audit failed: ' + str(comparisons))
        if not all(math.isclose(first, loss, rel_tol=1e-5, abs_tol=1e-6) for loss in (control, second)):
            raise RuntimeError(f'GPU resume next-loss mismatch: {first}, {second}')
        profile['resume_next_loss'] = [first, control, second]
        profile['resume_gradient_comparison'] = comparisons
        profile['resume_state_exact'] = True
        profile['resume_gradient_matches'] = True
        optimizer.zero_grad(set_to_none=True)
        del grad_first, grad_control, grad_second, optimizer, scheduler
        # Profile updates are discarded: A1 starts from official weights, never smoke/probe weights.
        load_official_countgdpp(model, checkpoint)
        optimizer, scheduler = make_optimizer()
        torch.manual_seed(config['seed']); random.seed(config['seed']); np.random.seed(config['seed'])
        profile['validation_probe'] = validation(0, profile=True)
        profile.update(peak_allocated_gib=torch.cuda.max_memory_allocated()/GIB,
                       peak_reserved_gib=torch.cuda.max_memory_reserved()/GIB, status='passed',
                       official_initialization_restored=True)
        write_json(args.output/'profile.json', profile)
        report['profile'] = {k: profile[k] for k in ('status', 'peak_allocated_gib', 'peak_reserved_gib', 'resume_gradient_matches')}
        print('Preflight/profile passed:', report['profile'], flush=True)
        del records; gc.collect(); torch.cuda.empty_cache()
    optimizer.zero_grad(set_to_none=True)
    save_training_checkpoint(args.output/'checkpoint_last.pth', model, optimizer, scheduler, identity=identity, progress=progress)
    if args.profile_only:
        report.update(status='profile_passed', exit_code=0, finished_utc=utc())
        return
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(progress['epoch'], config['epochs']):
        order = orders[epoch]
        start = progress['next_micro'] if epoch == progress['epoch'] else 0
        train_mode(); report.update(status='training', epoch=epoch+1)
        write_json(args.output/'report.json', report)
        window = []; window_start = time.monotonic()
        for offset in range(start, len(order)):
            source, index = order[offset]
            window.append(backward_sample(source, index, epoch, offset, config['gradient_accumulation']))
            if (offset+1) % config['gradient_accumulation']:
                continue
            if progress['global_update'] % config['save_every_updates'] == 0:
                report['last_gradient_audit'] = gradient_audit(model)
            norm = float(torch.nn.utils.clip_grad_norm_(trainable, config['gradient_clip'], error_if_nonfinite=True))
            lr_used = optimizer.param_groups[0]['lr']
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            progress.update(epoch=epoch, next_micro=offset+1, global_update=progress['global_update']+1, phase='train')
            report.update(status='training', optimizer_steps_completed=progress['global_update'],
                micro_steps_completed=epoch*len(order)+offset+1, epoch_micro_steps=offset+1,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/GIB,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/GIB, updated_utc=utc())
            row = {'epoch': epoch+1, 'update': progress['global_update'], 'samples': window,
                   'mean_loss': sum(v['loss'] for v in window)/len(window), 'gradient_norm': norm,
                   'learning_rate': lr_used, 'seconds': time.monotonic()-window_start}
            with (args.output/'training.jsonl').open('a') as f: f.write(json.dumps(row, allow_nan=False)+'\n')
            if progress['global_update'] <= 3 or progress['global_update'] % 10 == 0:
                print(f"TRAIN epoch={epoch+1} micro={offset+1}/{len(order)} update={progress['global_update']}/{total_updates} loss={row['mean_loss']:.5f} peak={report['peak_reserved_gib']:.2f} GiB", flush=True)
            write_json(args.output/'report.json', report)
            if progress['global_update'] == 1 or progress['global_update'] % config['save_every_updates'] == 0:
                save_training_checkpoint(args.output/'checkpoint_last.pth', model, optimizer, scheduler, identity=identity, progress=progress)
                report['checkpoint_update'] = progress['global_update']
            window = []; window_start = time.monotonic()
        progress.update(phase='validation', epoch=epoch, next_micro=len(order))
        save_training_checkpoint(args.output/'checkpoint_last.pth', model, optimizer, scheduler, identity=identity, progress=progress)
        report.update(status='validating', validation_images_completed=0); write_json(args.output/'report.json', report)
        metrics = validation(epoch+1)
        score = metrics['single']['macro_dataset_mae']
        if not math.isfinite(score): raise RuntimeError('Nonfinite validation score')
        improved = progress['best_score'] is None or score < progress['best_score']
        if improved: progress['best_score'] = score
        progress.update(epoch=epoch+1, next_micro=0, phase='train')
        save_training_checkpoint(args.output/'checkpoint_last.pth', model, optimizer, scheduler, identity=identity, progress=progress)
        if improved:
            destination = args.output/'checkpoint_best.pth'
            temporary = destination.with_suffix('.pth.tmp')
            if temporary.exists(): temporary.unlink()
            try: os.link(args.output/'checkpoint_last.pth', temporary)
            except OSError: shutil.copyfile(args.output/'checkpoint_last.pth', temporary)
            temporary.replace(destination)
            write_json(args.output/'best.json', {'epoch': epoch+1, 'update': progress['global_update'],
                'selection_metric': config['selection_metric'], 'score': score,
                'checkpoint_sha256': sha(destination), 'metrics': metrics})
        report.update(completed_epochs=epoch+1, best_score=progress['best_score'], last_validation=metrics)
        write_json(args.output/'report.json', report)
    report.update(status='completed', exit_code=0, finished_utc=utc())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/countgdpp_a1_v1.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-index', type=int, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--profile-only', action='store_true')
    args = parser.parse_args(); args.config = args.config.resolve(); args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'report.json').exists():
        parser.error('Use a new output directory, including when resuming')
    report = {'created_utc': utc(), 'status': 'starting'}; handle = None; exit_code = 0
    try:
        handle = gpu_lock(args, report)
        run(args, report)
    except BaseException as error:
        exit_code = 1
        report.update(status='oom_stopped' if isinstance(error, torch.cuda.OutOfMemoryError) else 'failed',
                      error=repr(error), traceback=traceback.format_exc(), exit_code=1, finished_utc=utc(),
                      recovery='No automatic retry or configuration reduction. Resume only from a saved boundary when explicitly requested.')
        traceback.print_exc()
    finally:
        write_json(args.output/'report.json', report)
        if handle: handle.close()
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
