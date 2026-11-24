"""Frozen SigLIP2 rich embeddings, cross-image batching, no label reads."""
import argparse,json,os,sys,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'env/step8_transformers451'));os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel,AutoProcessor
from src.semantic_expert_features import FrozenSemanticFeatures,views_for,FIXED,COMMON,chroma8
from src.semantic_runtime import sha,write,utc,inherited_lock

def body_view(row,size):
 x,y,xx,yy=row['box'];w,h=xx-x,yy-y
 if row['domain']=='relation':return [max(0,math.floor(x-.15*w)),max(0,math.floor(y-.8*h)),min(size[0],math.ceil(xx+.15*w)),min(size[1],math.ceil(yy+.05*h))]
 return [max(0,math.floor(x+.08*w)),max(0,math.floor(y+.3*h)),min(size[0],math.ceil(xx-.08*w)),min(size[1],math.ceil(yy-.08*h))]

class RichEngine(FrozenSemanticFeatures):
 supplied=None
 def encode(self,crops):
  if self.supplied is not None:
   value=self.supplied;self.supplied=None;assert len(value)==len(crops);return value
  return super().encode(crops)
 def cached(self,image,rows,embedding):
  self.supplied=embedding[:,:3].reshape(-1,embedding.shape[-1]);a=super().extract(image,rows);a['embedding']=embedding
  a['body_fixed']=np.stack([self.score(embedding[i,3:4],[[f'a photo of a {c} {r["noun"]}.',f'a {r["noun"]} with {c} bodywork.'] for c in FIXED])[0] for i,r in enumerate(rows)])
  crops=[image.crop(body_view(r,image.size)) for r in rows];a['body_chroma']=np.stack([chroma8(c) for c in crops])
  for c in crops:c.close()
  return a

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--limit',type=int);a=p.parse_args();out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True);assert not (out/'report.json').exists();lock=inherited_lock()
 rows=[json.loads(l) for l in (ROOT/a.manifest).open()];groups={}
 for r in rows:groups.setdefault(r['image_id'],[]).append(r)
 if a.limit:groups=dict(list(groups.items())[:a.limit])
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;free,total=torch.cuda.mem_get_info();assert free/2**30>=11;torch.cuda.set_per_process_memory_fraction(8*2**30/total)
 modelpath=ROOT/'checkpoints/step8/siglip2-base-patch16-384';identity=json.loads((modelpath/'download_identity.json').read_text())
 for n,h in identity['files_sha256'].items():assert sha(modelpath/n)==h
 report={'status':'loading','pid':os.getpid(),'started_utc':utc(),'images_expected':len(groups),'objects_expected':sum(map(len,groups.values())),'labels_read':False,'manifest_sha256':sha(ROOT/a.manifest),'source_sha256':sha(Path(__file__))};write(out/'report.json',report)
 model=AutoModel.from_pretrained(modelpath,local_files_only=True,torch_dtype=torch.float32).to('cuda:0').eval().requires_grad_(False);processor=AutoProcessor.from_pretrained(modelpath,local_files_only=True);engine=RichEngine(model,processor,32);cache={};ids=[];chunks=[];done=0;image_count=0;start=time.monotonic();pending=[];count=0
 def flush():
  nonlocal cache,ids
  if not ids:return
  path=out/f'features_{len(chunks):04d}.npz';np.savez_compressed(path,ids=np.asarray(ids),**{k:np.concatenate(v) for k,v in cache.items()});chunks.append({'file':path.name,'sha256':sha(path),'rows':len(ids)});write(out/'chunks.json',chunks);cache={};ids=[]
 def process():
  nonlocal pending,count,done,image_count
  crops=[]
  for image,rr in pending:
   for r in rr:
    for box in views_for(r['box'],image.size,r['domain'])+[body_view(r,image.size)]:crops.append(image.crop(box))
  if 'batch_profile' not in report:
   profile=[];sample=(crops*max(1,math.ceil(32/len(crops))))[:32]
   engine.batch=8;engine.encode(sample[:8]);torch.cuda.synchronize()
   for batch in [8,16,32]:
    engine.batch=batch;torch.cuda.reset_peak_memory_stats();t=time.monotonic();engine.encode(sample);torch.cuda.synchronize();profile.append({'batch':batch,'seconds_32_crops':time.monotonic()-t,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30})
   engine.batch=min(profile,key=lambda x:x['seconds_32_crops'])['batch'];report['batch_profile']=profile;report['selected_batch']=engine.batch;write(out/'report.json',report)
  embedding=engine.encode(crops).reshape(-1,4,model.config.vision_config.hidden_size)
  for crop in crops:crop.close()
  offset=0
  for image,rr in pending:
   arrays=engine.cached(image,rr,embedding[offset:offset+len(rr)]);offset+=len(rr);ids.extend(r['id'] for r in rr)
   for k,v in arrays.items():cache.setdefault(k,[]).append(v)
   done+=len(rr);image_count+=1;image.close()
  if len(ids)>=256:flush()
  report.update(status='extracting',images_completed=image_count,objects_completed=done,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,elapsed_seconds=time.monotonic()-start,updated_utc=utc());write(out/'report.json',report);print('EMBED',image_count,'/',len(groups),'objects',done,flush=True);pending=[];count=0
 for key,rr in sorted(groups.items()):
  assert sha(ROOT/rr[0]['image'])==rr[0]['image_sha256']
  with Image.open(ROOT/rr[0]['image']) as im:image=im.convert('RGB')
  pending.append((image,rr));count+=len(rr)
  if count>=32:process()
 if pending:process()
 flush();write(out/'metadata.json',{'scale':engine.scale,'bias':engine.bias,'common':COMMON,'model':identity['model'],'revision':identity['revision'],'dtype':'float32','source_sha256':sha(Path(__file__)),'view_order':['tight','context','inset','body_or_headroom'],'box_source':'from manifest; VCoR whole-image crop or A3 proposal, separately flagged in requests','labels_read':False});report.update(status='completed',exit_code=0,finished_utc=utc());write(out/'report.json',report);lock.close()
if __name__=='__main__':main()
