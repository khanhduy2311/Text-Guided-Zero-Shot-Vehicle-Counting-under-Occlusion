"""Deterministic clean/occluded source pairs, same geometry and labels as A2."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import deque
import multiprocessing
import torch
from .counting_data.loader import UnifiedCountingDataset
from .countgdpp_occlusion import occlude_source
from .a2_runtime import pack_arrays,unpack_arrays

_DATASETS=None
_CONFIG=None

def prepare_pair(datasets,config,source,index,epoch,offset,force=False):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config['seed']+epoch*1000003+offset)
        clean,target=datasets[source][index]
    image,mask,meta=occlude_source(clean,target,config['occlusion'],epoch=epoch,offset=offset,force=force)
    return clean,image,target,mask,meta


def initialize(root,config):
    global _DATASETS,_CONFIG
    torch.set_num_threads(1);_CONFIG=config
    _DATASETS={s:UnifiedCountingDataset(Path(root)/config['bundle']/f'source_train_{s}.jsonl',root,purpose='train',
        short_side=config['short_side'],max_size=config['max_size'],text_only=True,horizontal_flip_p=config['horizontal_flip_p']) for s in config['sources']}

def worker(task):return pack_arrays(prepare_pair(_DATASETS,_CONFIG,*task))

class PairPrefetch:
    def __init__(self,root,config):
        self.depth=config['num_workers']*2
        self.pool=ProcessPoolExecutor(max_workers=config['num_workers'],mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(str(root),config))
    def epoch(self,order,epoch,start):
        pending=deque();tasks=iter(enumerate(order[start:],start))
        def submit():
            try:offset,(source,index)=next(tasks)
            except StopIteration:return
            pending.append((offset,self.pool.submit(worker,(source,index,epoch,offset))))
        for _ in range(self.depth):submit()
        while pending:
            offset,future=pending.popleft();sample=unpack_arrays(future.result());submit();yield offset,sample
    def close(self):self.pool.shutdown(wait=True,cancel_futures=True)
