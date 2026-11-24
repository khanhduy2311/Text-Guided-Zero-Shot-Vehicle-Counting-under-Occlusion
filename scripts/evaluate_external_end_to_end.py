"""Select detector/semantic/type-gate combination on validation, then test once."""
import sys,json,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.semantic_runtime import write,sha,utc
from src.semantic_expert_calibration import load_features,binary_metrics
from src.semantic_expert_features import TYPE_TEXT,COMMON
from src.semantic_expert_policy import color_predictions,relation_predictions
from src.semantic_counting_v2 import iou
B=ROOT/'outputs/semantic_expanded_20261113';E=B/'e2e'
def match(objects,gt):
 if not objects or not gt:return []
 matrix=np.asarray([[iou(o['box'],g['box']) for g in gt] for o in objects]);a,b=linear_sum_assignment(np.where(matrix>=.5,1000+matrix,0),maximize=True);return [(int(i),int(j)) for i,j in zip(a,b) if matrix[i,j]>=.5]
def main():
 assert not (E/'report.json').exists();ids,arrays,metadata=load_features(E/'proposal_features');index={k:i for i,k in enumerate(ids)};requests=[json.loads(l) for l in (E/'semantic_requests.jsonl').open()];images=json.loads((E/'requests.json').read_text())['images'];ground={r['image_id']:r for r in json.loads((E/'ground_truth.json').read_text())};presence_labels=json.loads((E/'image_presence_labels.json').read_text())['labels'];groups={}
 for r in requests:groups.setdefault((r['backend'],r['image_id']),[]).append(r)
 ce=json.loads((B/'selected_color_expert.json').read_text());re=json.loads((B/'selected_relation_expert.json').read_text());scores={}
 for domain in ['vehicle','relation']:
  ix=np.asarray([index[r['id']] for r in requests if r['domain']==domain]);rows=[r for r in requests if r['domain']==domain];sub={k:v[ix] for k,v in arrays.items()};pred=color_predictions(sub,metadata,'vehicle',ce) if domain=='vehicle' else relation_predictions(sub,metadata,re)
  for method,(score,decision) in pred.items():
   if domain=='relation':score=score[:,None];decision=decision[:,None]
   for n,r in enumerate(rows):scores[(r['id'],method)]={'score':score[n],'decision':decision[n],'type':list(TYPE_TEXT)[int(sub['types'][n,0].argmax())]}
 def evaluate(backend,method,gate,domain,role,keep=False):
  confusion=np.zeros((8 if domain=='vehicle' else 1,3),int);errors=[];positive_errors=[];zero_errors=[];intervals=[];exact=[];loc=np.zeros(3,int);rows=[];image_truth=[];image_decision=[]
  for image in images:
   if image['domain']!=domain or image['role']!=role:continue
   before=confusion.copy()
   gt=ground[image['image_id']];objects=groups.get((backend,image['image_id']),[]);objects=[o for o in objects if gate=='none' or scores[(o['id'],method)]['type']==image['noun'] or (gate=='family' and image['noun']=='car' and scores[(o['id'],method)]['type'] in ['car','van','pickup'])];pairs=match(objects,gt['objects']);matchedp={i for i,j in pairs};matchedg={j for i,j in pairs};ignored={i for i,o in enumerate(objects) if i not in matchedp and any(iou(o['box'],b)>=.5 for b in gt['ignore_boxes'])};loc+=np.asarray([len(pairs),len(objects)-len(pairs)-len(ignored),len(gt['objects'])-len(pairs)]);n=confusion.shape[0];pcount=np.zeros(n,int)
   for i,o in enumerate(objects):
    if i not in ignored:pcount+=scores[(o['id'],method)]['decision'].astype(int)
   for i,j in pairs:
    y=np.asarray(gt['objects'][j]['target']);p=scores[(objects[i]['id'],method)]['decision'];confusion[:,0]+=(y==1)&p;confusion[:,1]+=(y==0)&p;confusion[:,2]+=(y==1)&~p
   for i,o in enumerate(objects):
    if i not in matchedp and i not in ignored:confusion[:,1]+=scores[(o['id'],method)]['decision']
   for j,o in enumerate(gt['objects']):
    if j not in matchedg:confusion[:,2]+=np.asarray(o['target'])==1
   y=np.asarray([o['target'] for o in gt['objects']],int).reshape(-1,n);lower=(y==1).sum(0);upper=lower+(y<0).sum(0);error=np.maximum(np.maximum(lower-pcount,pcount-upper),0);intervals.extend(error.tolist());known=lower==upper;errors.extend(abs(pcount[known]-lower[known]).tolist());exact.extend((pcount[known]==lower[known]).tolist());rows.append({'image_id':image['image_id'],'count':pcount.tolist(),'gt_lower':lower.tolist(),'gt_upper':upper.tolist(),'interval_error':error.tolist(),'object_ids':[o['id'] for o in objects]})
   rows[-1]['attribute_tp_fp_fn']=(confusion-before).tolist()
   positive_errors.extend(abs(pcount[known&(lower>0)]-lower[known&(lower>0)]).tolist());zero_errors.extend(abs(pcount[known&(lower==0)]-lower[known&(lower==0)]).tolist())
   if domain=='relation':
    truth=bool(presence_labels[image['image_id']]);decision=bool(pcount[0]);image_truth.append(truth);image_decision.append(decision);rows[-1]['image_presence_tp_fp_fn']=[int(truth and decision),int(not truth and decision),int(truth and not decision)];rows[-1]['official_image_presence_truth']=int(truth)
  den=2*confusion[:,0]+confusion[:,1]+confusion[:,2];per=np.divide(2*confusion[:,0],den,out=np.zeros(len(den)),where=den>0);s=confusion.sum(0);f1=2*s[0]/max(2*s[0]+s[1]+s[2],1);result={'images':len(rows),'macro_f1':float(per.mean()),'micro_f1':float(f1),'tp_fp_fn_per_target':confusion.tolist(),'per_target_f1':per.tolist(),'known_complete_image_query_count_mae':float(np.mean(errors)) if errors else None,'known_complete_image_query_count_exact':float(np.mean(exact)) if exact else None,'known_complete_image_queries':len(errors),'mean_distance_to_gt_count_interval':float(np.mean(intervals)),'interval_queries':len(intervals),'localization_tp_fp_fn':loc.tolist()}
  result.update(positive_known_query_count_mae=float(np.mean(positive_errors)) if positive_errors else None,positive_known_queries=len(positive_errors),zero_known_query_count_mae=float(np.mean(zero_errors)) if zero_errors else None,zero_known_queries=len(zero_errors))
  supported=(confusion[:,0]+confusion[:,2])>0;result['macro_f1']=float(per[supported].mean()) if supported.any() else 0.;result['macro_f1_supported_targets']=int(supported.sum());result['target_gt_positive']=(confusion[:,0]+confusion[:,2]).tolist()
  if domain=='relation':result['official_action_image_presence']=binary_metrics(image_truth,image_decision);result['count_warning']='HICO boxes identify interactions and can duplicate one motorcycle across riders/actions. Canonicalized counts and object F1 are exploratory only, not certified unique-vehicle counting.'
  if keep:result['images_detail']=rows
  return result
 candidates={};selected={}
 for domain in ['vehicle','relation']:
  candidates[domain]={}
  for backend in ['legacy','official','A3','official_hr']:
   for method in ['legacy_rule','fixed_rule','calibrated']:
    for gate in ['none','strict','family']:
     if domain=='relation' and gate=='family':continue
     key='/'.join([backend,method,gate]);candidates[domain][key]=evaluate(backend,method,gate,domain,'validation')
  # F1 first, then count interval error, then simpler gate. No test values used here.
  selected[domain]=max(candidates[domain],key=lambda k:((candidates[domain][k]['official_action_image_presence']['f1'] or 0) if domain=='relation' else candidates[domain][k]['macro_f1'],-candidates[domain][k]['mean_distance_to_gt_count_interval']))
 write(E/'selected_end_to_end.json',{'status':'frozen_before_test_scoring','selected':selected,'validation':candidates,'selected_utc':utc(),'code_sha256':sha(Path(__file__)),'features_sha256':sha(E/'proposal_features/chunks.json')})
 controls=[('legacy','legacy_rule','none'),('official','fixed_rule','strict'),('A3','fixed_rule','strict'),('official','calibrated','none'),('A3','calibrated','none')];report={'status':'completed','test_used_for_selection':False,'selected':selected,'test':{},'controls':{},'completed_utc':utc(),'scope':'Actual raw-image detector outputs plus matched FP32 semantic rules; legacy control is not bit-exact historical end-to-end code. HICO object counts derived from action-box canonicalization. Unknown OVAD attributes ignored; count intervals reported.'};table=[]
 for domain in ['vehicle','relation']:
  report['test'][domain]=evaluate(*selected[domain].split('/'),domain,'test',keep=True);report['controls'][domain]={}
  for args in controls:
   key='/'.join(args);r=evaluate(*args,domain,'test',keep=True);report['controls'][domain][key]=r;table.append({'domain':domain,'configuration':key,**{k:v for k,v in r.items() if k not in ['images_detail','per_target_f1','tp_fp_fn_per_target','localization_tp_fp_fn','official_action_image_presence','count_warning']},'image_presence_f1':r.get('official_action_image_presence',{}).get('f1')})
 comparisons={}
 for domain in ['vehicle','relation']:
  comparisons[domain]={};candidate=report['test'][domain]['images_detail']
  for baseline in ['legacy/legacy_rule/none','official/fixed_rule/strict','A3/fixed_rule/strict']:
   control=report['controls'][domain][baseline]['images_detail'];assert [r['image_id'] for r in candidate]==[r['image_id'] for r in control];key='attribute_tp_fp_fn' if domain=='vehicle' else 'image_presence_tp_fp_fn';stats=np.asarray([[a[key],b[key]] for a,b in zip(control,candidate)])
   if domain=='relation':stats=stats[:,:,None,:]
   supported=(stats[:,0,:,0]+stats[:,0,:,2]).sum(0)>0;rng=np.random.default_rng(20261114);deltas=[];count_deltas=[];count_difference=np.asarray([np.mean(b['interval_error'])-np.mean(a['interval_error']) for a,b in zip(control,candidate)])
   for start in range(0,10000,250):
    draw=rng.integers(0,len(stats),(250,len(stats)));s=stats[draw].sum(1);den=2*s[:,:,:,0]+s[:,:,:,1]+s[:,:,:,2];f=np.divide(2*s[:,:,:,0],den,out=np.zeros_like(den,dtype=float),where=den>0)[:,:,supported].mean(2);deltas.extend(f[:,1]-f[:,0]);count_deltas.extend(count_difference[draw].mean(1))
   comparisons[domain][baseline]={'selected_minus_baseline_f1_ci95':np.quantile(deltas,[.025,.975]).tolist(),'f1_scope':'attribute localization macroF1' if domain=='vehicle' else 'published action image presence F1','selected_minus_baseline_count_interval_distance_ci95':np.quantile(count_deltas,[.025,.975]).tolist(),'paired_image_bootstrap':10000,'count_warning':'HICO count interval metric exploratory; not unique vehicle GT.' if domain=='relation' else 'Unknown OVAD attributes produce GT intervals.'}
 report['paired_comparisons']=comparisons;write(E/'report.json',report)
 with (E/'object_semantics.jsonl').open('x') as f:
  for r in requests:f.write(json.dumps({**r,'methods':{m:{'score':scores[(r['id'],m)]['score'].tolist(),'decision':scores[(r['id'],m)]['decision'].astype(int).tolist(),'type':scores[(r['id'],m)]['type']} for m in ['legacy_rule','fixed_rule','calibrated']}})+'\n')
 with (E/'summary.csv').open('x') as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
 print(json.dumps({'selected':selected,'test':{d:{k:v for k,v in r.items() if k!='images_detail'} for d,r in report['test'].items()}},indent=2))
if __name__=='__main__':main()
