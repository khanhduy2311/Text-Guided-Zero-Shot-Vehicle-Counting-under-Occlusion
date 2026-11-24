"""Runtime-only A2 acceleration: bounded CPU prefetch and audited checkpoint migration."""
from concurrent.futures import ProcessPoolExecutor
from collections import deque
from collections.abc import Mapping
import multiprocessing
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from .counting_data.loader import UnifiedCountingDataset
from .countgdpp_occlusion import occlude_source
from .countgdpp_training import load_training_checkpoint, trees_equal, capture_rng

_WORKER_DATASETS = None
_WORKER_CONFIG = None


def pack_arrays(value):
    # Plain NumPy pickle over pipes avoids Docker's limited /dev/shm tensor IPC.
    if torch.is_tensor(value): return value.numpy()
    if isinstance(value, dict): return {k: pack_arrays(v) for k,v in value.items()}
    if isinstance(value, list): return [pack_arrays(v) for v in value]
    if isinstance(value, tuple): return tuple(pack_arrays(v) for v in value)
    return value


def unpack_arrays(value):
    if isinstance(value, np.ndarray): return torch.from_numpy(value)
    if isinstance(value, dict): return {k: unpack_arrays(v) for k,v in value.items()}
    if isinstance(value, list): return [unpack_arrays(v) for v in value]
    if isinstance(value, tuple): return tuple(unpack_arrays(v) for v in value)
    return value


def initialize_worker(root, config):
    global _WORKER_DATASETS, _WORKER_CONFIG
    torch.set_num_threads(1)
    _WORKER_CONFIG = config
    _WORKER_DATASETS = {s: UnifiedCountingDataset(Path(root)/config['bundle']/f'source_train_{s}.jsonl', root,
        purpose='train', short_side=config['short_side'],max_size=config['max_size'],text_only=True,
        horizontal_flip_p=config['horizontal_flip_p']) for s in config['sources']}


def prepare_sample(datasets, config, source, index, epoch, offset):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config['seed'] + epoch*1000003 + offset)
        image,target=datasets[source][index]
    image,_,meta=occlude_source(image,target,config['occlusion'],epoch=epoch,offset=offset)
    return image,target,meta


def worker_sample(task):
    return pack_arrays(prepare_sample(_WORKER_DATASETS,_WORKER_CONFIG,*task))


class SourcePrefetch:
    """Schedule-ordered, deterministic prefetch; never touches parent/CUDA RNG."""
    def __init__(self,root,config):
        workers=config['num_workers']
        if workers<1 or workers>4: raise ValueError('Expected 1–4 bounded CPU workers')
        self.depth=workers*2
        self.pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize_worker,initargs=(str(root),config))

    def epoch(self,order,epoch,start):
        pending=deque(); tasks=iter(enumerate(order[start:],start))
        def submit():
            try: offset,(source,index)=next(tasks)
            except StopIteration:return False
            pending.append((offset,self.pool.submit(worker_sample,(source,index,epoch,offset))));return True
        for _ in range(self.depth):submit()
        while pending:
            offset,future=pending.popleft();sample=unpack_arrays(future.result());submit()
            yield offset,sample

    def close(self):self.pool.shutdown(wait=True,cancel_futures=True)


RUNTIME_FIELDS = {'gradient_checkpointing','allocator_limit_gib','free_memory_reserve_gib','num_workers','cpu_threads','batch_size','gradient_accumulation','budget_note'}


def validate_runtime_migration(previous, current, root, sha):
    before,after=previous['config'],current['config']
    if before.get('batch_size',1)*before.get('gradient_accumulation',8) != after.get('batch_size',1)*after.get('gradient_accumulation',8):
        raise ValueError('Runtime migration changes effective batch')
    for key in before.keys()|after.keys():
        if key not in RUNTIME_FIELDS and before.get(key)!=after.get(key):
            raise ValueError('Runtime migration changes training protocol: '+key)
    for key in previous.keys()|current.keys():
        if key not in ('config','code_sha256') and previous.get(key)!=current.get(key):
            raise ValueError('Runtime migration identity mismatch: '+key)
    # Old source is immutable, including its runner/config. New runtime runner gets
    # its own identity; do not silently bless changed model/data/loss/augmentation.
    for name,digest in previous['code_sha256'].items():
        if sha(Path(root)/name)!=digest:raise ValueError('Historical code changed: '+name)
        if name in current['code_sha256'] and current['code_sha256'][name]!=digest:
            raise ValueError('Shared source changed: '+name)


def migrate_checkpoint(path,model,optimizer,scheduler,*,identity,root,sha,output,write_json):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state['format']!='countgdpp_a1_v1':raise ValueError('Unsupported checkpoint')
    validate_runtime_migration(state['identity'],identity,root,sha)
    progress=load_training_checkpoint(path,model,optimizer,scheduler,identity=state['identity'])
    for key,actual in [('model',model.state_dict()),('optimizer',optimizer.state_dict()),('scheduler',scheduler.state_dict()),('rng',capture_rng())]:
        if not trees_equal(actual,state[key]):raise ValueError('Migration failed exact '+key+' restore')
    write_json(Path(output)/'runtime_migration.json',dict(source_checkpoint=str(Path(path).relative_to(root)),
        source_sha256=sha(path),progress=progress,model_optimizer_scheduler_rng_exact=True,
        runtime_changes={key:{'before':state['identity']['config'].get(key),'after':identity['config'].get(key)}
            for key in RUNTIME_FIELDS if state['identity']['config'].get(key)!=identity['config'].get(key)},
        note='Same sample order, per-image objective, effective batch, LR/scheduler and restored RNG state. Physical batching changes padding and subsequent RNG consumption; not bitwise-equivalent training.'))
    return progress


def imagewise_losses(criterion,outputs,batch):
    """Preserve original mean-of-per-image objectives when physical batch >1."""
    from dataclasses import replace
    size=len(batch.targets)
    def slice_output(value,index):
        if torch.is_tensor(value):
            if value.ndim and value.shape[0]==size:return value[index:index+1]
            return value
        if isinstance(value,Mapping):return {k:slice_output(v,index) for k,v in value.items()}
        if isinstance(value,list):return [slice_output(v,index) for v in value]
        if isinstance(value,tuple):return tuple(slice_output(v,index) for v in value)
        return value
    result=[]
    for i in range(size):
        one=replace(batch,targets=batch.targets[i:i+1],captions=batch.captions[i:i+1],
            cap_list=batch.cap_list[i:i+1],exemplars=batch.exemplars[i:i+1],exemplar_labels=batch.exemplar_labels[i:i+1],
            token_ids=batch.token_ids[i:i+1],phrase_mask=batch.phrase_mask[i:i+1])
        result.append(criterion(slice_output(outputs,i),one)['loss_total'])
    return result
