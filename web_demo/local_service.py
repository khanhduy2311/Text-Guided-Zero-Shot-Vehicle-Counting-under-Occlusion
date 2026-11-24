"""Serialize GPU phases across branches and isolate incompatible Transformers versions."""
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from web_demo.query import parse_query, needs_category_verification
from web_demo.service import BASE, ROOT, JOBS, Service

GPU = threading.Lock()
NAMES = {'a3':'CountGD++ A3 + SigLIP2', 'original':'CountGD++ gốc',
         'sam3':'CountGD-Box + SAM 3', 'a3_sam3':'CountGD++ A3 + SAM 3'}

def python_for(stage):
    return os.getenv('COUNTLAB_'+stage.upper()+'_PYTHON', str(ROOT/'.demo-envs'/stage/'bin/python'))

def run_stage(stage, args, payload=None):
    command = [python_for(stage), '-m', *args]
    env = {**os.environ, 'HF_HUB_OFFLINE':'1', 'TOKENIZERS_PARALLELISM':'false'}
    result = subprocess.run(command, cwd=ROOT, env=env, check=True, timeout=1200, text=True,
                            input=json.dumps(payload) if payload else None, stdout=subprocess.PIPE)
    return result.stdout

class LocalService(Service):
    def __init__(self, branch):
        self.branch = branch
        self.name = NAMES[branch]
        self.infer_stage = self.name
        super().__init__()

    def parse_prompt(self, prompt):
        query = parse_query(prompt)
        if self.branch != 'a3':
            query.update(pipeline=self.name, route=self.branch)
        return query

    def bootstrap(self):
        self.font_bytes = (BASE/'static/assets/BeVietnamPro-Bold.ttf').read_bytes()
        for stage in (['detector','semantic'] if self.branch=='a3' else
                      ['detector'] if self.branch=='original' else ['detector','sam']):
            if not Path(python_for(stage)).is_file():
                raise RuntimeError('Missing Python environment: '+stage)
        if not (ROOT/'checkpoints/countlab/installed.json').is_file():
            raise RuntimeError('Run the checkpoint installer')
        self.status, self.stage = 'ready', 'Sẵn sàng'

    def infer(self, job):
        with GPU:
            if self.cancelled(job):
                return None
            return self.infer_locked(job)

    def infer_locked(self, job):
        query = job['query']
        source = ROOT/job['image_path']
        folder = JOBS/job['id']
        settings = dict(keep_on_gpu=False,optimize_crops=True,
                        settings_sha256=hashlib.sha256(b'local-sequential-v1').hexdigest())
        q = query if self.branch=='a3' else dict(kind='count',noun=(
            query['semantic_prompt'] if self.branch=='original' else query['noun']))
        payload = dict(job_id=job['id'],image=str(source),query=q,execution=settings)
        if self.branch == 'sam3':
            proposals = folder/'proposals.json'
            begin=time.monotonic()
            run_stage('detector',['Cubi.infer_sam3','proposals','--image',str(source),
                      '--checkpoint','checkpoints/countlab/countgd-box.pth','--noun',q['noun'],'--out',str(proposals)])
            rows=json.loads(proposals.read_text())
            detector=dict(seconds=time.monotonic()-begin,detector_prompts=[q['noun']],gpu_residency='local GPU',
                          checkpoint_sha256=json.loads((BASE/'weights.json').read_text())['release']['countgd-box.pth']['sha256'],
                          precision='float32',objects=[dict(box=b,detector_score=s) for b,s in zip(rows['boxes'],rows['scores'])])
        else:
            mode='original' if self.branch=='original' else 'detector'
            detector=json.loads(run_stage('detector',['web_demo.model_stage',mode],payload))
        if self.cancelled(job):
            return None
        semantic=None
        if self.branch in ('sam3','a3_sam3'):
            props=folder/'proposals.json'
            if self.branch=='a3_sam3':
                props.write_text(json.dumps(dict(image_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                     boxes=[o['box'] for o in detector['objects']],scores=[o['detector_score'] for o in detector['objects']])))
            start=time.monotonic()
            result=json.loads(run_stage('sam',['Cubi.infer_sam3','verify','--image',str(source),'--proposals',str(props),
                       '--sam3','checkpoints/sam3','--prompt',query['semantic_prompt']]).strip().splitlines()[-1])
            semantic=dict(seconds=time.monotonic()-start, objects=result['objects'], candidates=len(detector['objects']),
                          semantic_revision='cba430d22f6fdc3f06ad3841274ec7bb55885f2f',gpu_residency='local GPU',
                          policy={'version':self.name+' · IoU 0.30'},prompt_trace={'sam3':query['semantic_prompt']})
        elif self.branch=='a3' and (q['kind']!='count' or needs_category_verification(q)):
            semantic=json.loads(run_stage('semantic',['web_demo.model_stage','semantic'],{**payload,'objects':detector['objects']}))
        return detector,semantic,settings
