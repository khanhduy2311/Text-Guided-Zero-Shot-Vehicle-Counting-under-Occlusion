"""Balanced SigLIP2 contrasts, type verification, and vehicle-first rider inference."""
import sys,json,argparse,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'env/step8_transformers451'))
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch,numpy as np,transformers
from transformers import AutoModel,AutoProcessor
from PIL import Image
from src.semantic_runtime import local_path,write,sha,utc,inherited_lock
from src.semantic_counting import summarize_objects,box_valid
from src.semantic_counting_v2 import deduplicate,crop_box,rider_crop,rider_upper_crop,ranking_state,type_matches
COLORS=['green','black','blue','red','white','yellow','gray','silver','orange','brown']
TYPES={'car':['a photo of a car.','a photo of a passenger car or SUV.'],'van':['a photo of a van.','a photo of a passenger minivan.'],'pickup':['a photo of a pickup truck.','a photo of a pickup vehicle.'],'motorcycle':['a photo of a motorcycle.','a photo of a motor scooter.'],'bicycle':['a photo of a bicycle.','a photo of a pedal bicycle.'],'truck':['a photo of a truck.','a photo of a large cargo truck.'],'bus':['a photo of a bus.','a photo of a passenger bus.'],'person':['a photo of a person.','a photo of a pedestrian.'],'background':['a photo of an empty road.','a photo of a building or vegetation.']}
RELATIONS={'riding':['a photo of a person riding a motorcycle.','a photo of a person seated on a motorcycle.'],'parked':['a photo of a parked motorcycle without a rider.','a photo of an unattended motorcycle.'],'standing':['a photo of a person standing beside a motorcycle.','a photo of a person walking past a motorcycle.']}

def main():
 p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=local_path(a.output);request=json.loads(local_path(a.request).read_text());config=json.loads(local_path(request['queries']).read_text());queries=config['queries'];handle=inherited_lock();torch.set_num_threads(4)
 if (out/'results.json').exists():raise ValueError('Use a fresh output directory')
 if transformers.__version__!='4.51.3' or torch.cuda.get_device_capability()[0]<8:raise ValueError('Verified overlay/native BF16 required')
 free,total=torch.cuda.mem_get_info();assert free/2**30>=11;torch.cuda.set_per_process_memory_fraction(8*2**30/total)
 proposals=json.loads((out/'proposals.json').read_text());report={'status':'loading_model','started_utc':utc(),'pid':os.getpid()};write(out/'semantic_report.json',report);modelpath=ROOT/'checkpoints/step8/siglip2-base-patch16-384';identity=json.loads((modelpath/'download_identity.json').read_text())
 for n,h in identity['files_sha256'].items():assert sha(modelpath/n)==h
 model=AutoModel.from_pretrained(modelpath,local_files_only=True,torch_dtype=torch.bfloat16).to('cuda:0').eval().requires_grad_(False);processor=AutoProcessor.from_pretrained(modelpath,local_files_only=True);scale=float(model.logit_scale.exp().item());bias=float(model.logit_bias.item());text_cache={};results=[];started=time.monotonic();torch.cuda.reset_peak_memory_stats()
 def encode(crops):
  if not crops:return np.empty((0,768),dtype=np.float32)
  rows=[]
  with torch.inference_mode():
   for start in range(0,len(crops),16):
    pixels=processor(images=crops[start:start+16],return_tensors='pt')['pixel_values'].to(device='cuda:0',dtype=torch.bfloat16);v=model.get_image_features(pixel_values=pixels).float();rows.append((v/v.norm(dim=-1,keepdim=True)).cpu().numpy())
  return np.concatenate(rows)
 def classify(features,classes):
  labels=list(classes);key=tuple(s for label in labels for s in classes[label]);lengths=[len(classes[label]) for label in labels]
  if key not in text_cache:
   with torch.inference_mode():
    tok=processor(text=list(key),padding='max_length',max_length=64,truncation=True,return_tensors='pt').to('cuda:0');v=model.get_text_features(**tok).float();text_cache[key]=(v/v.norm(dim=-1,keepdim=True)).cpu().numpy()
  raw=features@text_cache[key].T;columns=[];offset=0
  for length in lengths:columns.append(raw[:,offset:offset+length].mean(axis=1));offset+=length
  cosine=np.stack(columns,axis=1);logits=cosine*scale+bias;shift=logits-logits.max(axis=1,keepdims=True);prob=np.exp(shift);prob/=prob.sum(axis=1,keepdims=True)
  return [{'label':labels[int(np.argmax(p))],'preference':dict(zip(labels,map(float,p))),'cosines':dict(zip(labels,map(float,c))),'preference_is_calibrated_probability':False} for p,c in zip(prob,cosine)]
 def crops_for(image,boxes):
  crops=[image.crop(box) for box in boxes];features=encode(crops)
  for im in crops:im.close()
  return features
 for image_index,record in enumerate(proposals['images']):
  assert sha(local_path(record['image']))==record['image_sha256']
  with Image.open(local_path(record['image'])) as im:image=im.convert('RGB')
  result={'id':record['id'],'image':record['image'],'image_sha256':record['image_sha256'],'queries':[]};cache={}
  for prompt in {q['object_prompt'] for q in queries}:
   original=record['objects_by_prompt'][prompt];assert all(box_valid(o['box'],image.size) for o in original);objects,removed=deduplicate(original);objects=[o for o in objects if min(o['box'][2]-o['box'][0],o['box'][3]-o['box'][1])>=12]
   if not objects:cache[prompt]=(objects,removed,[],np.empty((0,768)),[]);continue
   tight_boxes=[crop_box(o['box'],image.size) for o in objects];features=crops_for(image,tight_boxes);types=classify(features,TYPES);cache[prompt]=(objects,removed,types,features,tight_boxes)
  for q in queries:
   objects,removed,types,features,tight_boxes=cache[q['object_prompt']];target=q.get('target_type',q['object_prompt']);items=[];pairs=[];color_scores={};relation_scores={};upper_scores={}
   candidate_indices=[i for i,t in enumerate(types) if type_matches(t['label'],target)]
   if q['kind']=='attribute' and candidate_indices:
    color_classes={c:[f'a photo of a {c} car.',f'a car with {c} bodywork.'] for c in COLORS};scores=classify(features[candidate_indices],color_classes);color_scores=dict(zip(candidate_indices,scores))
   if q['kind']=='relation' and candidate_indices:
    boxes=[rider_crop(objects[i]['box'],image.size) for i in candidate_indices];rf=crops_for(image,boxes);scores=classify(rf,RELATIONS);relation_scores={i:{**s,'crop_box':box} for i,s,box in zip(candidate_indices,scores,boxes)}
    if q.get('target_color'):
     boxes=[rider_upper_crop(objects[i]['box'],image.size) for i in candidate_indices];uf=crops_for(image,boxes);classes={c:[f'a motorcycle rider wearing a {c} shirt.',f'a motorcycle rider wearing a {c} jacket.'] for c in COLORS};scores=classify(uf,classes);upper_scores={i:{**s,'crop_box':box} for i,s,box in zip(candidate_indices,scores,boxes)}
   for i,obj in enumerate(objects):
    typ=types[i];entry={**obj,'type_verification':typ,'crop_box':tight_boxes[i]}
    if not type_matches(typ['label'],target):status='rejected';reason='wrong_object_type'
    elif q['kind']=='class':status='matched';reason='verified_object_type'
    elif q['kind']=='attribute':
     scores=color_scores[i];entry['color_verification']=scores;status=ranking_state(list(scores['preference']),list(scores['preference'].values()),q['target_color'],.25);reason='balanced_color_ranking'
    else:
     rel=relation_scores[i];entry['relation_verification']=rel;status=ranking_state(list(rel['preference']),list(rel['preference'].values()),'riding',.4);reason='vehicle_centered_relation_not_person_detection_gate'
     if q.get('target_color') and status=='matched':
      col=upper_scores[i];entry['rider_color_verification']=col;cs=ranking_state(list(col['preference']),list(col['preference'].values()),q['target_color'],.25);status=cs;reason='riding_and_rider_upper_clothing'
    entry.update(status=status,reason=reason);items.append(entry)
   result['queries'].append({'id':q['id'],'kind':q['kind'],'query':q,**summarize_objects(items),'objects':items,'pairs':pairs,'duplicates_removed':removed,'association_policy':'Vehicle-centered headroom crop; no mandatory standalone person detection. Clothing uses estimated rider upper crop, not verified garment segmentation.'});print('V2',record['id'],q['id'],'count',sum(o['status']=='matched' for o in items),'unknown',sum(o['status']=='unknown' for o in items),flush=True)
  results.append(result);image.close();report.update(status='inferring',images_completed=image_index+1,images_expected=len(proposals['images']),updated_utc=utc(),peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30);write(out/'semantic_report.json',report)
 write(out/'results.json',{'schema_version':2,'status':'completed_inference_only','labels_used':False,'training_performed':False,'accuracy_evaluated':False,'threshold_calibrated_for_color_relations':False,'engine':'balanced_contrasts_vehicle_centered_v2','policy':{'color_relative_preference_min':.25,'relation_relative_preference_min':.4,'minimum_side_px':12,'batch_size':16,'type_gate':'top1 among9 object types; car family car/van/pickup','color_templates':'Two identical templates per color; no tight/context AND','relation':'adaptive headroom crop, optional person not required','learned_logit_scale':scale,'learned_logit_bias':bias,'preference_is_calibrated_probability':False},'model':identity['model'],'revision':identity['revision'],'detector_provenance':proposals['provenance'],'images':results,'limitations':['Development revision after inspecting source-validation failures; original benchmark remains historical.','Preference scores are relative to prompt set, not calibrated probabilities.','Vehicle-centered relation and upper-clothing crops remain approximate.','Counting and localization can disagree; inspect FP/FN and images.']})
 report.update(status='completed',exit_code=0,finished_utc=utc(),elapsed_seconds=time.monotonic()-started);write(out/'semantic_report.json',report);handle.close()
if __name__=='__main__':main()
