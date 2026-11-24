"""Deterministic A1 sampling and complete optimizer-boundary checkpoints."""
import math
import random
from pathlib import Path

import numpy as np
import torch

SOURCES = ('fsc147', 'visdrone', 'ua_detrac')
DATASETS = {'fsc147': 'FSC147', 'visdrone': 'VisDrone', 'ua_detrac': 'UA_DETRAC'}


def training_pools(records, capacity=900):
    pools, exclusions = {}, []
    for source in SOURCES:
        pools[source] = []
        for i, r in enumerate(records[source]):
            if r['role'] != 'source_train' or r['dataset'] != DATASETS[source]:
                raise ValueError('A1 sampling requires approved source training records')
            if r['count'] > capacity:
                exclusions.append({'source': source, 'index': i, 'id': r['id'],
                                   'count': r['count'], 'reason': 'above_query_capacity_excluded_in_A1_v1'})
            else:
                pools[source].append(i)
        if not pools[source]:
            raise ValueError('Empty training source')
    return pools, exclusions


def epoch_schedule(pools, seed, epoch):
    """50/25/25 draws, complete coverage, shuffled cycles without replacement.

    One balanced epoch covers every eligible vehicle image and oversamples FSC147.
    Every eight samples contain exactly four FSC147/two VisDrone/two UA-DETRAC.
    """
    rng = random.Random(seed + 1000003 * epoch)
    n = max(len(pools['visdrone']), len(pools['ua_detrac']), math.ceil(len(pools['fsc147']) / 2))
    n += n % 2  # optimizer accumulation of eight ends exactly at epoch boundary
    draws = {}
    for source, amount in zip(SOURCES, (2*n, n, n)):
        values = []
        while len(values) < amount:
            cycle = list(pools[source])
            rng.shuffle(cycle)
            values.extend(cycle)
        draws[source] = values[:amount]
    order = []
    for i in range(n):
        block = [('fsc147', draws['fsc147'][2*i]), ('fsc147', draws['fsc147'][2*i+1]),
                 ('visdrone', draws['visdrone'][i]), ('ua_detrac', draws['ua_detrac'][i])]
        rng.shuffle(block)
        order.extend(block)
    return order


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def capture_rng():
    numpy_state = np.random.get_state()
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'numpy': [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]]}


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))


def trees_equal(a, b):
    """Exact state comparison independent of tensor device, for resume audits."""
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a.detach().cpu(), b.detach().cpu())
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(trees_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return isinstance(b, type(a)) and len(a) == len(b) and all(trees_equal(x, y) for x, y in zip(a, b))
    return a == b


def save_training_checkpoint(path, model, optimizer, scheduler, *, identity, progress):
    """Caller must have zeroed gradients at an optimizer boundary."""
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('Checkpoint must be saved at a cleared optimizer boundary')
    state = {'format': 'countgdpp_a1_v1', 'identity': identity, 'progress': progress,
             'model': cpu_tree(model.state_dict()), 'optimizer': cpu_tree(optimizer.state_dict()),
             'scheduler': scheduler.state_dict(), 'rng': capture_rng()}
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def load_training_checkpoint(path, model, optimizer, scheduler, *, identity):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['format'] != 'countgdpp_a1_v1' or state['identity'] != identity:
        raise ValueError('Resume identity mismatch; keep original code/config/data/runtime')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    restore_rng(state['rng'])
    return state['progress']


def learning_rate_factor(step, warmup, total):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    ratio = min(1., (step - warmup) / max(1, total - warmup))
    return .1 + .9 * .5 * (1. + math.cos(math.pi * ratio))
