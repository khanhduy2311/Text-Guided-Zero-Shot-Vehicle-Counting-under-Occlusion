"""Frozen four-arm semantic benchmark: original/trained direct text and base proposals."""
import sys,argparse,json,gc,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.semantic_runtime import local_path,write,sha,utc,inherited_lock
from scripts.run_countgdpp_step7 import torch,Image,SLConfig,build_groundingdino,nested_tensor_from_tensor_list,infer_step7,ms_deform_attn
from src.step7_inference import project_candidates
from src.countgdpp_checkpoint import load_official_countgdpp

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();out=local_path(args.output);plan=json.loads((out/'plan.json').read_text());handle=inherited_lock();torch.set_num_threads(4)
 for name,digest in plan['input_sha256'].items():
  if sha(ROOT/name)!=digest:raise ValueError('Frozen input changed: '+name)
 if torch.cuda.get_device_capability()[0]<8 or ms_deform_attn._C is None:raise ValueError('Native BF16/CUDA extension required')
 free,total=torch.cuda.mem_get_info()
 if free/2**30<9:raise ValueError('Need6allocator+3reserve GiB')
 torch.cuda.set_per_process_memory_fraction(6*2**30/total)
 cpath=ROOT/'configs/step7_v1.json';c=json.loads(cpath.read_text());c['tile_size']=2**31
 references=plan['checkpoints'];records={};raw=[];report={'status':'loading','pid':os.getpid(),'started_utc':utc(),'jobs_completed':0};write(out/'detector_report.json',report);started=time.monotonic()
 for backend in ['official','A3']:
  reference=references[backend];path=ROOT/reference['path'];assert sha(path)==reference['sha256']
  cfg=SLConfig.fromfile(str(ROOT/c['model_config']));cfg.device='cuda:0';cfg.text_encoder_type=str(ROOT/'CountGD/checkpoints/bert-base-uncased');cfg.use_checkpoint=cfg.use_transformer_ckpt=False
  model,_,_=build_groundingdino(cfg)
  if backend=='official':load_official_countgdpp(model,path)
  else:
   state=torch.load(path,map_location='cpu',weights_only=True,mmap=True);assert state['format']=='countgdpp_a1_v1' and state['identity']['config']['variant']=='A3'
   for name,digest in state['identity']['code_sha256'].items():
    if name.startswith('CountGDPlusPlus/') or name in ('src/countgdpp_inference.py','src/countgdpp_checkpoint.py'):assert sha(ROOT/name)==digest
   model.load_state_dict(state['model'],strict=True);del state
  gc.collect();model.to('cuda:0').eval().requires_grad_(False);torch.cuda.reset_peak_memory_stats()
  prompts=sorted(set([q['direct_prompt'] for q in plan['queries']]+['car','motorcycle']+(['person'] if backend=='official' else [])))
  for k,r in enumerate(plan['images']):
   assert sha(ROOT/r['image'])==r['sha256']
   with Image.open(ROOT/r['image']) as im:image=im.convert('RGB')
   for j,prompt in enumerate(prompts):
    outputs,metadata=infer_step7(model,image,prompt,nested_tensor_from_tensor_list,torch.device('cuda:0'),c)
    group=project_candidates(outputs['filtered_pseudo3'],(0,0,*image.size),image.size,c['box_threshold'],c['text_threshold'])
    objects=[{'id':f'{backend}_i{k:02d}_p{j:02d}_q{int(q):04d}','box':box,'detector_score':score,'query_index':int(q),'detector_variant':backend} for q,box,score in zip(group['query_indices'],group['boxes_xyxy'],group['scores'])]
    record={'image_id':r['id'],'image':r['image'],'backend':backend,'prompt':prompt,'objects':objects,'inference':metadata};raw.append(record);records[(backend,r['id'],prompt)]=objects
    with (out/'detector_predictions.jsonl.partial').open('a') as f:f.write(json.dumps(record)+'\n')
    report.update(status='inferring',backend=backend,image_id=r['id'],jobs_completed=len(raw),jobs_expected=len(plan['images'])*13,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,updated_utc=utc());write(out/'detector_report.json',report)
    print('Detected',backend,r['id'],prompt,len(objects),flush=True)
   image.close()
  del model;gc.collect();torch.cuda.empty_cache()
 (out/'detector_predictions.jsonl.partial').rename(out/'detector_predictions.jsonl')
 for backend in ['official','A3']:
  dest=out/(backend+'_siglip2');dest.mkdir(exist_ok=True)
  images=[]
  for r in plan['images']:
   images.append({'id':r['id'],'image':r['image'],'image_sha256':r['sha256'],'size_wh':[640,480],'objects_by_prompt':{p:records[(('official' if p=='person' else backend),r['id'],p)] for p in ['car','motorcycle','person']}})
  write(dest/'proposals.json',{'schema_version':1,'provenance':{'checkpoints':references,'prompt_backends':{'car':backend,'motorcycle':backend,'person':'official'},'source_mode':'filtered_pseudo3','image_annotations_used':False,'threshold':.23,'prompts':['car','motorcycle','person']},'images':images})
 report.update(status='completed',exit_code=0,elapsed_seconds=time.monotonic()-started,finished_utc=utc());write(out/'detector_report.json',report);handle.close()
if __name__=='__main__':main()
