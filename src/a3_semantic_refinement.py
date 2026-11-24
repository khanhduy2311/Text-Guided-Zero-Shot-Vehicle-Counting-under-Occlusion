"""Proposal-aware A3 refinement. Pure feature/scoring functions shared with runtime."""
import json
from pathlib import Path
import numpy as np
from src.semantic_expert_features import COMMON,CANONICAL,TYPE_TEXT,feature_matrix,preference
from src.semantic_expert_calibration import load_features,predict,binary_metrics
from src.semantic_counting_v2 import iou
from scripts.evaluate_external_end_to_end import match
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'outputs/a3_semantic_dev_20261114';OLD=ROOT/'outputs/semantic_expanded_20261113'
def additional_features(arrays,rows,scale):
 p=preference(arrays['types'],scale);cos=arrays['types'];extra=[]
 for r in rows:
  x,y,x2,y2=r['box'];w,h=r['size_wh'];score=np.clip(r['detector_score'],1e-5,1-1e-5);extra.append([np.log(score/(1-score)),np.log(max((x2-x)*(y2-y)/(w*h),1e-8)),(x+x2)/(2*w),(y+y2)/(2*h)])
 return np.concatenate([cos[:,0],p[:,0],np.asarray(extra)],1)
def target_features(arrays,rows,metadata,color=None,relation=False):
 return np.concatenate([feature_matrix(arrays,None if relation else CANONICAL.index(color),relation=relation,scale=metadata['scale']),additional_features(arrays,rows,metadata['scale'])],1)
def gate_mask(arrays,domain,gate,heads,rows,metadata):
 t=arrays['types'][:,0].argmax(1);target=list(TYPE_TEXT).index('car' if domain=='vehicle' else 'motorcycle')
 if gate=='none':return np.ones(len(rows),bool)
 if gate=='strict':return t==target
 if gate=='family' and domain=='vehicle':return np.isin(t,[0,1,2])
 if gate=='learned':return predict(heads['validity'],additional_features(arrays,rows,metadata['scale']))>=.5
 return t==target
def infer_scores(arrays,rows,metadata,policy,heads):
 domain=policy['domain'];scale=metadata['scale'];n=len(rows)
 if not n:return np.zeros((0,8 if domain=='vehicle' else 1)),np.zeros((0,8 if domain=='vehicle' else 1),bool)
 kind=policy['kind']
 if domain=='vehicle':
  if kind=='learned':score=np.stack([predict(heads['colors'][c],target_features(arrays,rows,metadata,color=c)) for c in COMMON],1)
  else:
   view=policy['view'];cos=arrays['fixed'][:,view] if view in [0,2] else arrays['fixed'].mean(1);p=preference(cos,scale);labels=['green','black','blue','red','white','yellow','gray','silver','orange','brown'];score=np.stack([p[:,[i for i,c in enumerate(labels) if c==color or color=='gray' and c=='silver']].sum(1) for color in COMMON],1)
   # Independent attributes rather than forced top1; validation chooses threshold.
 else:
  if kind=='pooled':
   rp=preference(arrays['relation'][:,1],scale);mask=gate_mask(arrays,domain,'strict',heads,rows,metadata);decision=(rp[:,0]>=.4)&(rp.argmax(1)==0)&mask;keys,x=image_matrix(arrays,rows,metadata);image_scores=dict(zip(keys,predict(heads['image_relation'],x)));accept=np.asarray([image_scores[r['image_id']]>=policy['threshold'] for r in rows]);return rp[:,0,None],(decision&accept)[:,None]
  if kind=='learned':score=predict(heads['relation'],target_features(arrays,rows,metadata,relation=True))[:,None]
  else:
   view=policy['view'];p=preference(arrays['relation'][:,view] if view in [0,1,2] else arrays['relation'].mean(1),scale);score=p[:,0,None]
 mask=gate_mask(arrays,domain,policy['gate'],heads,rows,metadata);decision=(score>=np.asarray(policy['threshold']))&mask[:,None]
 return score,decision
def image_matrix(arrays,rows,metadata):
 group={}
 for i,r in enumerate(rows):group.setdefault(r['image_id'],[]).append(i)
 tp=preference(arrays['types'],metadata['scale'])[:,0,list(TYPE_TEXT).index('motorcycle')];rel=preference(arrays['relation'],metadata['scale']);features=[]
 for key,ix in group.items():
  vals=[]
  for view in range(3):
   joint=tp[ix]*rel[ix,view,0];vals.extend([float(joint.max()),float(np.sort(joint)[-3:].mean()),float((tp[ix]*(rel[ix,view,0]-rel[ix,view,2])).max()),float(joint.mean())])
  vals.extend([np.log1p(len(ix)),np.log1p((arrays['types'][ix,0].argmax(1)==3).sum()),max((rows[i]['box'][2]-rows[i]['box'][0])*(rows[i]['box'][3]-rows[i]['box'][1])/(rows[i]['size_wh'][0]*rows[i]['size_wh'][1]) for i in ix),max(rows[i]['detector_score'] for i in ix)]);features.append(vals)
 return list(group),np.asarray(features)
def evaluate_policy(arrays,rows,metadata,policy,heads,images,gt,presence,role):
 score,decision=infer_scores(arrays,rows,metadata,policy,heads);mask=gate_mask(arrays,policy['domain'],'strict' if policy['kind']=='pooled' else policy['gate'],heads,rows,metadata);ix=np.flatnonzero(mask);return metrics([rows[i] for i in ix],decision[ix],images,gt,presence,policy['domain'],role)
def load_development():
 ids,a,meta=load_features(B/'train_features');req={r['id']:r for r in map(json.loads,(B/'train_features_requests.jsonl').open())};rows=[req[i] for i in ids];oldids,oldarrays,om=load_features(OLD/'e2e/proposal_features');oldreq={r['id']:r for r in map(json.loads,(OLD/'e2e/semantic_requests.jsonl').open())};oldimages=json.loads((OLD/'e2e/requests.json').read_text())['images'];roles={r['image_id']:r['role'] for r in oldimages};ix=[i for i,k in enumerate(oldids) if oldreq[k]['backend']=='A3'];rows += [{**oldreq[oldids[i]],'role':roles[oldreq[oldids[i]]['image_id']]} for i in ix];arrays={k:np.concatenate([v,oldarrays[k][ix]]) for k,v in a.items()};assert meta['scale']==om['scale'];images=json.loads((B/'train_request.json').read_text())['images']+oldimages;gt={r['image_id']:r for r in json.loads((B/'train_ground_truth.json').read_text())+json.loads((OLD/'e2e/ground_truth.json').read_text())};presence={r['id']:r['raw_image_onboard'] for r in json.loads((ROOT/'data/semantic_external_v1/hico_expert/images.json').read_text())};return rows,arrays,meta,images,gt,presence
def training_labels(rows,gt,presence):
 validity=[];targets=[]
 for r in rows:
  g=gt[r['image_id']];overlaps=[iou(r['box'],o['box']) for o in g['objects']];best=int(np.argmax(overlaps)) if overlaps else None;mx=max(overlaps,default=0);ignore=any(iou(r['box'],b)>=.3 for b in g['ignore_boxes']);v=1 if mx>=.5 else (0 if mx<.3 and not ignore else -1);validity.append(v)
  if r['domain']=='vehicle':y=g['objects'][best]['target'] if v==1 else ([0]*8 if v==0 else [-1]*8)
  elif not presence[r['image_id']]:y=[0]
  else:
   positive=max([o for o,gto in zip(overlaps,g['objects']) if gto['target'][0]==1],default=0);y=[1 if positive>=.5 else (0 if mx<.1 else -1)]
  targets.append(y)
 return np.asarray(validity),targets
def metrics(rows,decision,images,gt,presence,domain,role):
 byimage={}
 for i,r in enumerate(rows):byimage.setdefault(r['image_id'],[]).append(i)
 n=8 if domain=='vehicle' else 1;conf=np.zeros((n,3),int);details=[];errors=[];truth=[];pred=[]
 for image in images:
  if image['domain']!=domain or image['role']!=role:continue
  key=image['image_id'];indices=byimage.get(key,[]);objs=[rows[i] for i in indices];g=gt[key];pairs=match(objs,g['objects']);pi={i for i,j in pairs};gi={j for i,j in pairs};ignored={i for i,o in enumerate(objs) if i not in pi and any(iou(o['box'],b)>=.5 for b in g['ignore_boxes'])};before=conf.copy();counts=np.zeros(n,int)
  for i,r in enumerate(objs):
   if i not in ignored:counts+=decision[indices[i]].astype(int)
  for i,j in pairs:
   y=np.asarray(g['objects'][j]['target']);p=decision[indices[i]];conf[:,0]+=(y==1)&p;conf[:,1]+=(y==0)&p;conf[:,2]+=(y==1)&~p
  for i in range(len(objs)):
   if i not in pi and i not in ignored:conf[:,1]+=decision[indices[i]]
  for j,gto in enumerate(g['objects']):
   if j not in gi:conf[:,2]+=np.asarray(gto['target'])==1
  y=np.asarray([o['target'] for o in g['objects']],int).reshape(-1,n);lo=(y==1).sum(0);hi=lo+(y<0).sum(0);known=lo==hi;errors.extend(abs(counts[known]-lo[known]).tolist());row={'image_id':key,'count':counts.tolist(),'gt_lower':lo.tolist(),'gt_upper':hi.tolist(),'confusion':(conf-before).tolist()}
  if domain=='relation':truth.append(presence[key]);pred.append(int(counts[0]>0));row.update(presence_target=truth[-1],presence_predicted=pred[-1])
  details.append(row)
 support=conf[:,0]+conf[:,2]>0;den=2*conf[:,0]+conf[:,1]+conf[:,2];f=np.divide(2*conf[:,0],den,out=np.zeros(n),where=den>0);m={'images':len(details),'macro_f1':float(f[support].mean()) if support.any() else 0,'per_target_f1':f.tolist(),'confusion':conf.tolist(),'known_count_mae':float(np.mean(errors)) if errors else None,'known_queries':len(errors),'images_detail':details}
 if domain=='relation':m['presence']=binary_metrics(truth,pred)
 return m
