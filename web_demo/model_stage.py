"""Persistent JSON-line GPU workers, isolated to preserve frozen import environments."""
import os,sys,json,time,gc,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
original = sys.argv[1] == 'original'
kind = 'detector' if original else sys.argv[1]
os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
protocol=sys.stdout
sys.stdout=sys.stderr  # Model-library print statements must never corrupt RPC.
import faulthandler,signal
faulthandler.enable(file=sys.stderr)
faulthandler.register(signal.SIGUSR1,file=sys.stderr,all_threads=True)
print('Worker diagnostics ready:',kind,'python',sys.version.split()[0],'local_prefix',sys.prefix.startswith('/tmp/'),flush=True)
def send(data):protocol.write(json.dumps(data,ensure_ascii=False,allow_nan=False)+'\n');protocol.flush()

def offloaded(model,runner):
 # Service serializes phases and chooses residency from a measured GPU budget.
 import torch
 def memory():
  return {'idle_reserved_mib':torch.cuda.memory_reserved()/2**20,'gpu_free_mib':torch.cuda.mem_get_info()[0]/2**20}
 def release():
  model.to('cpu');torch.cuda.synchronize();torch.cuda.empty_cache()
  return {**memory(),'gpu_residency':'CPU offload'}
 def run(req):
  if req.get('control')=='memory':return memory()
  if req.get('control')=='offload':return release()
  start=time.monotonic()
  free,_=torch.cuda.mem_get_info()
  minimum_gib=7 if kind=='detector' and torch.cuda.get_device_capability()[0]<8 else 3
  if free<minimum_gib*2**30:
   from web_demo.open_text import PromptLengthError
   raise PromptLengthError('GPU vừa thiếu bộ nhớ dự phòng. Tác vụ đã dừng trước bước tiếp theo; hãy thử lại sau.')
  keep=req.get('execution',{}).get('keep_on_gpu',False);succeeded=False
  try:
   model.to('cuda:0');torch.cuda.synchronize();uploaded=time.monotonic()
   result=runner(req);torch.cuda.synchronize();computed=time.monotonic();succeeded=True
  finally:
   if not keep or not succeeded:release()
  end=time.monotonic()
  result.update(seconds=end-start,gpu_residency='GPU resident' if keep else 'CPU offload',
                model_upload_seconds=uploaded-start,model_compute_seconds=computed-uploaded,
                model_offload_seconds=end-computed,**memory())
  return result
 return run

def detector():
 print("Detector: importing frozen libraries",flush=True)
 from scripts.run_countgdpp_step7 import torch,Image,SLConfig,build_groundingdino,nested_tensor_from_tensor_list,infer_step7,ms_deform_attn
 from src.semantic_runtime import sha
 from src.a3_full_detector import proposal_views
 from src.step7_inference import project_candidates
 print("Detector: imports ready; verifying checkpoint",flush=True)
 torch.set_num_threads(4)

 native_bf16=torch.cuda.get_device_capability()[0]>=8
 _,total=torch.cuda.mem_get_info();torch.cuda.set_per_process_memory_fraction((2.5 if native_bf16 else 6)*2**30/total)
 c=json.loads((ROOT/'configs/step7_v1.json').read_text());c['tile_size']=2**31
 if original:
  identity=json.loads((ROOT/'configs/pretrained_countgdpp_official.json').read_text())
  path=ROOT/identity['path']; checkpoint_sha=identity['sha256']
 else:
  path=ROOT/'checkpoints/countlab/a3.pth';checkpoint_sha=release_hash('a3.pth')
 cfg=SLConfig.fromfile(str(ROOT/c['model_config']));cfg.device='cuda:0';cfg.text_encoder_type=str(ROOT/'CountGD/checkpoints/bert-base-uncased');cfg.use_checkpoint=cfg.use_transformer_ckpt=False
 print('Detector: building model',flush=True)
 model,_,_=build_groundingdino(cfg)
 if original:
  from src.countgdpp_checkpoint import load_official_countgdpp
  load_official_countgdpp(model,path)
 else:
  state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)

  model.load_state_dict(state['model'],strict=True);del state
 gc.collect();model.eval().requires_grad_(False)
 if not native_bf16:
  from web_demo.precision import wrap_fp32
  model=wrap_fp32(model).eval().requires_grad_(False)
  print('Detector: pre-Ampere uses FP32; frozen research unchanged',flush=True)
 def run(req):
  if 'image_b64' in req:
   import base64,io
   source=io.BytesIO(base64.b64decode(req['image_b64'],validate=True))
  else:source=ROOT/req['image']
  with Image.open(source) as im:image=im.convert('RGB')
  query=req['query'];start=time.monotonic();torch.cuda.reset_peak_memory_stats()
  if query['kind']=='count':
   outputs,meta=infer_step7(model,image,query['noun'],nested_tensor_from_tensor_list,torch.device('cuda:0'),c)
   g=project_candidates(outputs['filtered_pseudo3'],(0,0,*image.size),image.size,c['box_threshold'],c['text_threshold'])
   objects=[{'id':f'object-{i+1}','box':box,'detector_score':float(score),'query_index':int(q)} for i,(box,score,q) in enumerate(zip(g['boxes_xyxy'],g['scores'],g['query_indices']))]
  elif query['kind'] in ['open_color','open_text']:
   from web_demo.open_text import detect
   objects,prompts=detect(model,image,query,nested_tensor_from_tensor_list,req.get('policy'))
  else:
   objects=proposal_views(model,image,query['noun'],nested_tensor_from_tensor_list,False)
   for i,obj in enumerate(objects):obj.update(id=f'object-{i+1}',image_id=req['job_id'],domain='vehicle' if query['kind']=='color' else 'relation',noun=query['noun'])
  image.close();torch.cuda.synchronize();return {'detector_prompts':prompts if query['kind'] in ['open_color','open_text'] else [query['noun']],'objects':objects,'seconds':time.monotonic()-start,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'checkpoint_sha256':checkpoint_sha,'precision':'bfloat16' if native_bf16 else 'float32'}
 return offloaded(model,run)

def semantic():
 print("Semantic: importing frozen libraries",flush=True)
 import numpy as np
 import torch
 from PIL import Image
 from transformers import AutoModel,AutoProcessor
 from scripts.extract_full_semantic_embeddings import RichEngine,body_view
 from src.semantic_expert_features import views_for,COMMON
 from src.full_semantic_policy import prepare_scores,decisions_for
 from src.a3_color_cv import predict
 from src.semantic_runtime import sha
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 _,total=torch.cuda.mem_get_info();torch.cuda.set_per_process_memory_fraction(2.5*2**30/total)
 print('Semantic: imports ready; verifying weights',flush=True)
 path=ROOT/'checkpoints/step8/siglip2-base-patch16-384';identity=json.loads((ROOT/'web_demo/policies/siglip2.json').read_text())
 for name,h in identity['files_sha256'].items():assert sha(path/name)==h
 model=AutoModel.from_pretrained(path,local_files_only=True,torch_dtype=torch.float32).eval().requires_grad_(False);processor=AutoProcessor.from_pretrained(path,local_files_only=True);engine=RichEngine(model,processor,16)
 policies=ROOT/'web_demo/policies'
 def read_policy(name):return json.loads((policies/(name+'.json')).read_text())
 selection=read_policy('selection');cv=read_policy('color_cv_selection');verifiers=read_policy('verifiers')['heads'];old=read_policy('previous_selection');state=torch.load(ROOT/'checkpoints/countlab/semantic-head.pt',map_location='cpu',weights_only=True)
 def run(req):
  rows=req['objects'];query=req['query'];start=time.monotonic();torch.cuda.reset_peak_memory_stats()
  if not rows:
   from web_demo.open_text import load_policy,policy_sha
   policy=req.get('policy') or load_policy()
   return {'objects':[],'candidates':0,'seconds':0,'peak_reserved_gib':torch.cuda.memory_reserved()/2**30,'semantic_revision':identity['revision'],**({'policy':policy,'policy_sha256':policy_sha(policy)} if query['kind'] in ['open_text','open_color','count'] else {})}
  if 'image_b64' in req:
   import base64,io
   source=io.BytesIO(base64.b64decode(req['image_b64'],validate=True))
  else:source=ROOT/req['image']
  with Image.open(source) as im:image=im.convert('RGB')
  if query['kind'] in ['open_color','open_text','count']:
   from web_demo.open_text import semantic as open_semantic
   try:result=open_semantic(engine,image,rows,query,req.get('policy'),req.get('execution'))
   finally:image.close()
   torch.cuda.synchronize();return {**result,'seconds':time.monotonic()-start,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'semantic_revision':identity['revision']}
  crops=[image.crop(b) for r in rows for b in views_for(r['box'],image.size,r['domain'])+[body_view(r,image.size)]]
  embedding=engine.encode(crops).reshape(-1,4,model.config.vision_config.hidden_size)
  for c in crops:c.close()
  arrays=engine.cached(image,rows,embedding);image.close();metadata={'scale':engine.scale,'bias':engine.bias};scores=prepare_scores(rows,arrays,metadata,state,verifiers)
  if query['kind']=='color':
   rr,p,d,ix=predict(rows,scores,cv);j=COMMON.index(query['color']);threshold=cv['selected']['policy']['threshold'][j]
  else:
   policy=selection['domains']['relation']['policy'];rr,p,d,ix=decisions_for(rows,arrays,metadata,scores,policy,old);j=0;threshold=.4
  objects=[]
  for i,r in enumerate(rr):
   if bool(d[i,j]):objects.append({'id':r['id'],'box':r['box'],'detector_score':float(r['detector_score']),'semantic_score':float(p[i,j]),'semantic_threshold':float(threshold)})
  torch.cuda.synchronize();return {'objects':objects,'candidates':len(rr),'seconds':time.monotonic()-start,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'semantic_revision':identity['revision']}
 return offloaded(model,run)

def release_hash(name):
 return json.loads((ROOT/'web_demo/weights.json').read_text())['release'][name]['sha256']

if __name__=='__main__':
 request=json.load(sys.stdin)
 runner=detector() if kind=='detector' else semantic()
 import torch
 with torch.inference_mode():
  result=runner(request)
 send(result)
