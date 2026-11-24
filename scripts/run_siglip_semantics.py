"""Uncalibrated zero-shot attributes and person-object context on detector boxes."""
import sys,json,argparse,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'env/step8_transformers451'))
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch,numpy as np,transformers
from transformers import AutoModel,AutoProcessor
from PIL import Image
from src.semantic_runtime import local_path,write,sha,utc,inherited_lock
from src.semantic_counting import validate_queries,decision,associate,reduce_pairs,summarize_objects,box_valid

def main():
 p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--output',required=True);args=p.parse_args();out=local_path(args.output);request=json.loads(local_path(args.request).read_text());config=json.loads(local_path(request['queries']).read_text());queries=validate_queries(config);policy=config.get('policy',{})
 margin=policy.get('cosine_margin',.02);expansion=policy.get('context_expansion',.15);minimum=policy.get('minimum_side_px',12)
 handle=inherited_lock();torch.set_num_threads(4)
 if transformers.__version__!='4.51.3' or torch.cuda.get_device_capability()[0]<8:raise ValueError('Verified transformers4.51.3/native BF16 runtime required')
 free,total=torch.cuda.mem_get_info()
 if free/2**30<11:raise ValueError('Need 8GiB allocator +3GiB reserve')
 torch.cuda.set_per_process_memory_fraction(8*2**30/total)
 proposals=json.loads((out/'proposals.json').read_text());report={'status':'loading_model','started_utc':utc(),'pid':os.getpid()};write(out/'semantic_report.json',report)
 modelpath=ROOT/'checkpoints/step8/siglip2-base-patch16-384';identity=json.loads((modelpath/'download_identity.json').read_text())
 for name,digest in identity['files_sha256'].items():
  if sha(modelpath/name)!=digest:raise ValueError('SigLIP2 checkpoint/processor changed')
 model=AutoModel.from_pretrained(modelpath,local_files_only=True,torch_dtype=torch.bfloat16).to('cuda:0').eval().requires_grad_(False);processor=AutoProcessor.from_pretrained(modelpath,local_files_only=True)
 text_cache={};results=[];started=time.monotonic();torch.cuda.reset_peak_memory_stats()
 def evaluate(image,box,positive,negative,context):
  x1,y1,x2,y2=box
  if min(x2-x1,y2-y1)<minimum:return {'status':'unknown','reason':'crop_too_small','box':box}
  dx=(x2-x1)*context/2;dy=(y2-y1)*context/2;b=[max(0,int(np.floor(x1-dx))),max(0,int(np.floor(y1-dy))),min(image.width,int(np.ceil(x2+dx))),min(image.height,int(np.ceil(y2+dy)))]
  key=tuple(positive+negative)
  with torch.inference_mode():
   if key not in text_cache:
    tokens=processor(text=list(key),padding='max_length',max_length=64,truncation=True,return_tensors='pt').to('cuda:0');v=model.get_text_features(**tokens).float();text_cache[key]=v/v.norm(dim=-1,keepdim=True)
   crop=image.crop(b);pixels=processor(images=[crop],return_tensors='pt')['pixel_values'].to(device='cuda:0',dtype=torch.bfloat16);crop.close();v=model.get_image_features(pixel_values=pixels).float();v=v/v.norm(dim=-1,keepdim=True);scores=(v@text_cache[key].T)[0].cpu().tolist()
  pos=max(scores[:len(positive)]);neg=max(scores[len(positive):]);return {'status':decision(pos,neg,margin),'positive_cosine':pos,'negative_cosine':neg,'margin':pos-neg,'crop_box':b,'all_text_cosines':scores,'score_is_probability':False}
 for record in proposals['images']:
  path=local_path(record['image'])
  if sha(path)!=record['image_sha256']:raise ValueError('Image changed between detector and semantic stages')
  with Image.open(path) as im:image=im.convert('RGB')
  image_result={'id':record['id'],'image':record['image'],'image_sha256':record['image_sha256'],'queries':[]}
  for q in queries:
   objects=record['objects_by_prompt'][q['object_prompt']]
   if any(not box_valid(o['box'],image.size) for o in objects):raise ValueError('Invalid detector box')
   pairs=[]
   if q['kind']=='class':items=[{**o,'status':'matched','reason':'class_only_semantics_bypassed'} for o in objects]
   elif q['kind']=='attribute':
    items=[]
    for obj in objects:
     tight=evaluate(image,obj['box'],q['positive'],q['negative'],0);context=evaluate(image,obj['box'],q['positive'],q['negative'],expansion)
     status=tight['status'] if tight['status']==context['status'] else 'unknown'
     items.append({**obj,'status':status,'tight':tight,'context':context,'reason':'tight_context_agreement' if tight['status']==context['status'] else 'tight_context_disagreement'})
   else:
    subjects=record['objects_by_prompt'][q['subject_prompt']]
    associations,ambiguous,unassigned=associate(subjects,objects,q['relation'],policy.get('association_margin',.1))
    for index,pair in enumerate(associations):
     relation=evaluate(image,pair['box'],q['positive'],q['negative'],expansion);checks=[relation['status']];subject=None
     if 'subject_positive' in q:
      sb=pair['subject_box'];focus=[sb[0],sb[1]+.2*(sb[3]-sb[1]),sb[2],sb[1]+.65*(sb[3]-sb[1])] if q.get('subject_region')=='upper_body' else sb
      subject=evaluate(image,focus,q['subject_positive'],q['subject_negative'],0);checks.append(subject['status'])
     status='rejected' if 'rejected' in checks else ('matched' if all(x=='matched' for x in checks) else 'unknown')
     pairs.append({**pair,'pair_id':f"{q['id']}_pair_{index:04d}",'status':status,'relation_semantics':relation,'subject_attribute':subject})
    items=reduce_pairs(objects,pairs,ambiguous)
   item={'id':q['id'],'kind':q['kind'],'query':q,**summarize_objects(items),'objects':items,'pairs':pairs}
   if q['kind']=='relation':item.update(ambiguous_object_ids=sorted(ambiguous),unassigned_subject_ids=unassigned,association_policy='image-plane geometric candidates + joint crop semantics; not verified physical interaction')
   image_result['queries'].append(item);print('Semantic',record['id'],q['id'],'count',item['count'],'unknown',item['unknown_count'],flush=True)
  image.close();results.append(image_result);report.update(status='inferring',images_completed=len(results),updated_utc=utc(),peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30);write(out/'semantic_report.json',report)
 result={'schema_version':1,'status':'completed_inference_only','labels_required':False,'labels_used':False,'training_performed':False,'accuracy_evaluated':False,'threshold_calibrated_for_color_relations':False,'policy':policy,'model':identity['model'],'revision':identity['revision'],'detector_provenance':proposals['provenance'],'images':results,'limitations':['Cosine similarity is not probability.','Defaults are uncalibrated; unknown crops and ambiguous associations are not counted.','Missed detector objects cannot be recovered by this filtering stage.','Each object ID is counted at most once per query, even with multiple people.','No accuracy/MAE claim without verified labels.']}
 write(out/'results.json',result);report.update(status='completed',exit_code=0,finished_utc=utc(),elapsed_seconds=time.monotonic()-started,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30);write(out/'semantic_report.json',report);handle.close()
if __name__=='__main__':main()
