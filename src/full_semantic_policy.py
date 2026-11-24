"""Shared proposal scoring, deduplication and object/pair decisions."""
import json
from pathlib import Path
import numpy as np
import torch
from src.full_semantic_heads import head_predict,common_colors,relation_features,validity_features,pair_features
from src.semantic_expert_features import preference,FIXED,COMMON
from src.semantic_expert_calibration import predict
from src.a3_full_detector import suppress
from src.a3_semantic_refinement import infer_scores,metrics,match
from src.semantic_counting_v2 import iou,deduplicate

def prepare_scores(rows,arrays,metadata,color_state,verifiers,pair_bundle=None):
 domain=rows[0]['domain'] if rows else 'vehicle';types=arrays['types'][:,0].argmax(1);result={'types':types,'validity':{c:predict(h,validity_features(arrays)) for c,h in verifiers[domain]['validity'].items()}}
 if domain=='vehicle':
  result['trained_color']=common_colors(head_predict(color_state,arrays))
  if '_proposal_adaptation' in color_state:result['adapted_color']=common_colors(head_predict(color_state['_proposal_adaptation'],arrays))
  for name,cosine in [('multiview_color',arrays['fixed'].mean(1)),('body_color',arrays['body_fixed'])]:
   p=preference(cosine,metadata['scale']);result[name]=np.stack([p[:,[i for i,c in enumerate(FIXED) if c==color or color=='gray' and c=='silver']].sum(1) for color in COMMON],1)
 else:
  rp=preference(arrays['relation'][:,1],metadata['scale']);result['fixed_relation']=rp[:,0];result['fixed_relation_decision']=(rp[:,0]>=.4)&(rp.argmax(1)==0)
  result['object_relation']={c:predict(h,relation_features(arrays)) for c,h in verifiers['relation']['object'].items()};result['pair_relation']={};result['pair_available']=np.zeros(len(rows),bool)
  if pair_bundle and pair_bundle[0]:
   pr,pa=pair_bundle;px=pair_features(pa,pr);byimage={}
   for j,r in enumerate(pr):byimage.setdefault(r['image_id'],[]).append(j)
   association=[]
   for i,r in enumerate(rows):
    candidates=[(iou(r['box'],pr[j]['vehicle_box']),j) for j in byimage.get(r['image_id'],[])];best=max(candidates,default=(0,-1));association.append(best[1] if best[0]>=.65 else -1)
   association=np.asarray(association);valid=association>=0;result['pair_available']=valid
   for c,h in verifiers['relation']['pair'].items():
    p=predict(h,px);values=np.zeros(len(rows));values[valid]=p[association[valid]];result['pair_relation'][c]=values
 return result

def select_boxes(rows,scores,policy):
 chosen=[];byimage={}
 for i,r in enumerate(rows):byimage.setdefault(r['image_id'],[]).append(i)
 for image,indices in byimage.items():
  accepted=[];quality=[]
  if policy.get('legacy_dedup'):
   eligible=[i for i in indices if rows[i]['source_view']==policy['views'] and rows[i]['detector_score']>=policy['detector_threshold']]
   kept,_=deduplicate([rows[i] for i in eligible]);keep_ids={r['id'] for r in kept};indices=[i for i in eligible if rows[i]['id'] in keep_ids]
  for i in indices:
   r=rows[i];view=r['source_view'];views=policy['views']
   if views=='baseline640' and view!='baseline640':continue
   if views=='full800' and view!='full800':continue
   if views=='multiscale' and view=='baseline640':continue
   if r['detector_score']<policy['detector_threshold']:continue
   gate=policy['gate'];domain=r['domain'];target=0 if domain=='vehicle' else 3
   if gate=='strict' and scores['types'][i]!=target:continue
   if gate=='family' and scores['types'][i] not in [0,1,2]:continue
   validity=scores['validity'][policy.get('validity_C','0.1')][i]
   if gate=='learned' and validity<policy['validity_threshold']:continue
   accepted.append(i);quality.append(r['detector_score']*(validity if policy.get('quality')=='verified' else 1.))
  local=list(range(len(accepted))) if policy.get('legacy_dedup') else suppress([rows[i] for i in accepted],quality,policy['nms_iou'])
  chosen.extend(accepted[j] for j in local)
 return np.asarray(sorted(chosen),dtype=int)

def decisions_for(rows,arrays,metadata,scores,policy,old_selection=None):
 ix=select_boxes(rows,scores,policy['boxes']);domain=policy['domain'];rr=[rows[i] for i in ix];kind=policy['semantic']['kind'];semantic=policy['semantic']
 if domain=='vehicle':
  if kind=='trained':p=scores['trained_color'][ix]
  elif kind=='adapted':p=scores['adapted_color'][ix]
  elif kind=='adapted_blend':p=semantic['weight']*scores['adapted_color'][ix]+(1-semantic['weight'])*scores['multiview_color'][ix]
  elif kind=='blend':p=semantic['weight']*scores['trained_color'][ix]+(1-semantic['weight'])*scores['multiview_color'][ix]
  elif kind=='body':p=.5*scores['body_color'][ix]+.5*scores['multiview_color'][ix]
  elif kind=='multiview':p=scores['multiview_color'][ix]
  elif kind=='fixed':
   from src.a3_semantic_release import release_predictions
   aa={k:v[ix] for k,v in arrays.items()};p,d=release_predictions(aa,rr,metadata,{'domain':'vehicle','kind':'fixed','gate':'strict'},old_selection['domains']['vehicle']['heads']);return rr,p,d,ix
  d=p>=semantic['threshold']
 else:
  if kind=='fixed':p=scores['fixed_relation'][ix,None];d=scores['fixed_relation_decision'][ix,None]
  elif kind=='old_image':
   all_ix=select_boxes(rows,scores,{**policy['boxes'],'gate':'none'});all_rows=[rows[i] for i in all_ix];aa={k:v[all_ix] for k,v in arrays.items()};entry=old_selection['domains']['relation'];pp,dd=infer_scores(aa,all_rows,metadata,entry['policy'],entry['heads']);lookup={int(i):j for j,i in enumerate(all_ix)};jj=[lookup[int(i)] for i in ix];p,d=pp[jj],dd[jj]
  else:
   p=scores['object_relation'][semantic['C']][ix].copy()
   if kind=='pair_blend':
    valid=scores['pair_available'][ix]
    if valid.any():
     q=scores['pair_relation'][semantic['pair_C']][ix];p[valid]=(1-semantic['pair_weight'])*p[valid]+semantic['pair_weight']*q[valid]
   p=p[:,None];d=p>=semantic['threshold']
 return rr,p,d,ix

def evaluate(rows,arrays,metadata,scores,policy,images,gt,presence,role,old_selection):
 rr,p,d,ix=decisions_for(rows,arrays,metadata,scores,policy,old_selection);m=metrics(rr,d,images,gt,presence,policy['domain'],role);byimage={}
 for r in rr:byimage.setdefault(r['image_id'],[]).append(r)
 loc=np.zeros(3,int);errors=[]
 for image in images:
  if image['domain']!=policy['domain'] or image['role']!=role:continue
  g=gt[image['image_id']];found=byimage.get(image['image_id'],[]);pairs=match(found,g['objects']);matched={i for i,j in pairs};ignored={i for i,r in enumerate(found) if i not in matched and any(iou(r['box'],box)>=.5 for box in g['ignore_boxes'])};loc += [len(pairs),len(found)-len(pairs)-len(ignored),len(g['objects'])-len(pairs)];errors.append(abs(len(found)-len(ignored)-len(g['objects'])))
 m['localization_tp_fp_fn']=loc.tolist();den=2*loc[0]+loc[1]+loc[2];m['localization_f1']=float(2*loc[0]/den) if den else 0.;m['all_object_count_mae']=float(np.mean(errors)) if errors else None
 return m

def controls(domain):
 fixed={'domain':domain,'boxes':{'views':'baseline640','detector_threshold':.23,'gate':'strict','nms_iou':.85,'quality':'detector','legacy_dedup':True},'semantic':{'kind':'fixed'}}
 old={'domain':domain,'boxes':{'views':'full800' if domain=='vehicle' else 'baseline640','detector_threshold':.23,'gate':'family' if domain=='vehicle' else 'strict','nms_iou':.85,'quality':'detector','legacy_dedup':True},'semantic':{'kind':'multiview','threshold':.5} if domain=='vehicle' else {'kind':'old_image'}}
 return {'A3_fixed':fixed,'A3_previous_release':old}
