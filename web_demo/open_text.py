"""Versioned, label-free demo extension. Does not change frozen research recipes.

Relative softmax values are ranking scores, not calibrated correctness probabilities.
Inference receives no labels/reference counts or image-specific selection rules.
QA-informed type counterexamples are documented; this is not a held-out benchmark.
"""
import re,hashlib,json
from pathlib import Path

def policy_sha(policy):
 return hashlib.sha256(json.dumps(policy,sort_keys=True).encode()).hexdigest()

def load_policy():
 """Local deployment configuration; captured once per job by Service."""
 p=json.loads(Path(__file__).with_name('open_text_policy.json').read_text())
 if p['version']!='open_text_v2':raise ValueError('Unknown serving policy')
 for key in ['detector_min','nms_iou','color_preference_min','description_preference_min','query_vs_noun_min','type_preference_min']:
  if not 0<=p[key]<=1:raise ValueError('Invalid serving threshold: '+key)
 if not 1<=p['max_candidates']<=200:raise ValueError('Invalid candidate limit')
 for key in ['color_views','description_views','type_views']:
  v=p[key]
  if len(v)!=3 or min(v)<0 or abs(sum(v)-1)>1e-6:raise ValueError('Invalid crop weights')
 guard=p.get('independent_type_guard')
 if guard and (guard.get('view') not in ['tight','tight_and_inset'] or not 0<guard.get('preference_min',0)<=1 or
               not isinstance(guard.get('nouns'),list) or any(n not in TYPE_NAMES for n in guard['nouns'])):
  raise ValueError('Invalid independent category guard')
 if guard and (not 0<=guard.get('center_inset_fraction',.1)<.5 or
               guard.get('center_policy','require_target') not in ['require_target','conflicting_vehicle_veto'] or
               not 0<guard.get('center_preference_min',.5)<=1):raise ValueError('Invalid center category check')
 return p
COLORS=['black','blue','brown','gray','green','orange','pink','red','beige','purple','white','yellow']
TYPE_NAMES=['car','van','pickup truck','motorcycle','bicycle','truck','bus','train','tractor','trailer','vehicle wheel','person','background']
# A pickup is a truck. The converse is not true for narrow categories: matching
# "car" alone must not certify "taxi", "police car" or "SUV".
FAMILIES={'truck':['pickup truck']}

class PromptLengthError(ValueError):pass

def caption(text):
 # CountGD treats periods/question marks as phrase delimiters. Keep all words.
 return ' '.join(re.sub(r'[.!?;\n\r]',' ',text).split())

def templates(text):return [f'a photo of {text}.',f'an image of {text}.']

def contrasts(prompt,noun):
 """Counterfactuals change a single property; every other condition stays intact."""
 result=[]
 for color in COLORS:
  if re.search(r'\b'+color+r'\b',prompt,re.I):
   result.extend(re.sub(r'\b'+color+r'\b',other,prompt,flags=re.I) for other in COLORS if other!=color)
 from web_demo.query import OPEN_COLORS
 for color in OPEN_COLORS.values():
  if re.search(r'\b'+re.escape(color)+r'\b',prompt,re.I):
   result.extend(re.sub(r'\b'+re.escape(color)+r'\b',other,prompt,flags=re.I) for other in OPEN_COLORS.values() if other!=color)
 pairs=[('with a rider','without a rider'),('with rider','without rider'),('with people','without people'),
        ('parked','moving'),('empty','occupied'),('large','small'),('damaged','undamaged')]
 for a,b in pairs:
  for source,target in [(a,b),(b,a)]:
   if re.search(r'\b'+re.escape(source)+r'\b',prompt,re.I):result.append(re.sub(r'\b'+re.escape(source)+r'\b',target,prompt,flags=re.I))
 if noun!='vehicle':
  for other in TYPE_NAMES:
   if other not in [noun,'background']:
    changed=re.sub(r'\b'+re.escape(noun)+r's?\b',other,prompt,flags=re.I)
    if changed!=prompt:result.append(changed)
 result+=['an empty road without vehicles','a person or a building, without a vehicle']
 return list(dict.fromkeys(x for x in result if x.lower()!=prompt.lower()))[:40]

def category_baseline(prompt,noun):
 from web_demo.query import VEHICLES
 return VEHICLES.get(noun,noun) if re.search(r'\b(?:xe|ô tô|phương tiện)\b',prompt,re.I) else 'a '+noun

def validate_tokens(engine,groups):
 strings=[s for g in groups for s in g]
 tokens=engine.processor.tokenizer(strings,truncation=False,padding=False)['input_ids']
 limit=engine.model.config.text_config.max_position_embeddings
 if any(len(t)>limit for t in tokens):
  raise PromptLengthError('Mô tả vượt giới hạn token của SigLIP2. Hãy viết ngắn hơn, giữ các đặc điểm quan trọng.')

def detect(model,image,query,nested_factory,policy=None):
 from src.a3_full_detector import proposal_views,suppress
 policy=policy or load_policy()
 rows=[];prompts=[query['noun']]
 description=caption(query['semantic_prompt'])
 # Negative text must not ask the detector to localize the excluded color.
 if not query.get('color_negated') and description.lower()!=query['noun'].lower():prompts.append(description)
 per_prompt=[]
 for prompt in prompts:
  group=[]
  for row in proposal_views(model,image,prompt,nested_factory,policy.get('proposal_tiles',False)):
   if row['detector_score']>=policy['detector_min']:
    group.append({**row,'id':f'proposal-{len(group)+1}','domain':'vehicle','noun':query['noun'],'detector_prompt':prompt})
  keep=suppress(group,[r['detector_score'] for r in group],threshold=policy['nms_iou'])
  per_prompt.append([group[i] for i in keep])
 # Keep alternative localizations across prompts until semantic verification.
 # Round-robin prevents a dense noun-only pass from exhausting the whole budget.
 for rank in range(max(map(len,per_prompt),default=0)):
  for group in per_prompt:
   if rank<len(group):rows.append(group[rank])
 rows=rows[:policy['max_candidates']]
 return [{**r,'id':f'object-{j+1}'} for j,r in enumerate(rows)],prompts

def semantic(engine,image,rows,query,policy=None,execution=None):
 import numpy as np
 from src.semantic_expert_features import preference,views_for
 from src.semantic_counting_v2 import crop_box
 from src.a3_full_detector import suppress
 policy=policy or load_policy();sha=policy_sha(policy)
 if not rows:return {'objects':[],'candidates':0,'policy':policy,'policy_sha256':sha}
 from web_demo.crop_runtime import active_views,encode_views,audit_cosines
 boxes=[]
 for row in rows:
  tight,_,inset=views_for(row['box'],image.size,'vehicle')
  guard=policy.get('independent_type_guard',{})
  if query['noun'] in guard.get('nouns',[]) and guard.get('center_policy')=='conflicting_vehicle_veto':
   import math
   x,y,xx,yy=row['box'];pad=guard['center_inset_fraction'];w=xx-x;h=yy-y
   inset=[max(0,math.floor(x+pad*w)),max(0,math.floor(y+pad*h)),min(image.width,math.ceil(xx-pad*w)),min(image.height,math.ceil(yy-pad*h))]
  context=crop_box(row['box'],image.size,.4)
  boxes.append([tight,inset,context])
 emb,crop_trace=encode_views(engine,image,boxes,active_views(policy,query),(execution or {}).get('optimize_crops',False))
 noun=query['noun'];names=list(TYPE_NAMES)
 if noun not in names:names.insert(0,noun)
 from src.semantic_expert_features import TYPE_TEXT
 type_prompts={**TYPE_TEXT,'pickup truck':TYPE_TEXT['pickup'],'background':['an empty road.','a building or vegetation.'],'trailer':['a photo of a truck trailer.','a trailer without a truck cab.'],'vehicle wheel':['a photo of a vehicle wheel.','a photo of tires.']}
 groups=[type_prompts.get(n,templates('a '+n)) for n in names]
 validate_tokens(engine,groups)
 type_cos=engine.score(emb.reshape(-1,emb.shape[-1]),groups).reshape(len(rows),3,-1)
 neutral_type_cos=type_cos
 conditioned_type_cos=None
 if query['kind']=='open_color' and not query.get('color_negated'):
  # Bind color to the candidate's category. A partly visible colored car can
  # have an ambiguous silhouette; a bare-type top-1 is not the whole query.
  color=query['color']
  aliases={'car':['car','passenger car or SUV'],'van':['van','passenger minivan'],
           'pickup truck':['pickup truck','pickup vehicle'],'truck':['truck','cargo truck'],
           'motorcycle':['motorcycle','motor scooter'],'bus':['bus','passenger bus']}
  conditional_groups=[]
  for n in names:
   if n in ['background','person','vehicle wheel']:g=type_prompts.get(n,templates('a '+n))
   elif n=='trailer':g=[f'a photo of a {color} truck trailer.',f'a {color} trailer without a truck cab.']
   else:g=[f'a photo of a {color} {a}.' for a in aliases.get(n,[n,n])]
   conditional_groups.append(g)
  validate_tokens(engine,conditional_groups)
  conditioned_type_cos=engine.score(emb.reshape(-1,emb.shape[-1]),conditional_groups).reshape(len(rows),3,-1)
  if policy.get('color_type_mode','neutral')=='conditioned':type_cos=conditioned_type_cos
 type_weights=[1,0,0] if query['kind']=='count' else policy['type_views']
 type_scores=preference(np.einsum('nvc,v->nc',type_cos,type_weights),engine.scale);best_type=type_scores.argmax(-1)
 accepted=set([noun,*policy.get('type_families',FAMILIES).get(noun,[])])
 if noun=='vehicle':accepted=set(names)-{'person','background','vehicle wheel'}
 valid=np.array([names[i] in accepted for i in best_type])&(type_scores.max(-1)>=policy['type_preference_min'])
 # Color-conditioned text and context can let a neighboring motorcycle
 # certify a car crop. Require a separate, neutral tight-crop type decision
 # for the configured categories. These preferences are not probabilities of
 # correctness. Ambiguous crops are withheld rather than certified by color.
 guard=policy.get('independent_type_guard',{});guard_scores=None;guard_valid=None;inset_scores=None
 if noun in guard.get('nouns',[]):
  guard_scores=preference(neutral_type_cos[:,0,:],engine.scale);guard_best=guard_scores.argmax(-1)
  guard_valid=np.array([names[i] in accepted for i in guard_best])&(guard_scores.max(-1)>=guard['preference_min'])
  if guard.get('view')=='tight_and_inset':
   inset_scores=preference(neutral_type_cos[:,1,:],engine.scale);inset_best=inset_scores.argmax(-1)
   if guard.get('center_policy')=='conflicting_vehicle_veto':
    # A motorcycle center often contains its rider. A person/uncertain center
    # is not evidence against the bike; a clear car/bus/etc. is a contradiction.
    conflict=set(guard['conflicting_types'])-accepted
    guard_valid &= ~(np.array([names[i] in conflict for i in inset_best])&(inset_scores.max(-1)>=guard['center_preference_min']))
   else:guard_valid &= np.array([names[i] in accepted for i in inset_best])&(inset_scores.max(-1)>=policy['type_preference_min'])
  valid &= guard_valid
 if query['kind']=='count':
  score=guard_scores.max(-1) if guard_scores is not None else type_scores.max(-1)
  threshold=guard['preference_min'] if guard_scores is not None else policy['type_preference_min']
  extra={'verification':'category_only','accepted_types':sorted(accepted)}
 elif query['kind']=='open_color':
  groups=[[f'a photo of a {c} {noun}.',f'a {noun} with {c} bodywork.'] for c in COLORS]
  validate_tokens(engine,groups)
  color_cos=engine.score(emb.reshape(-1,emb.shape[-1]),groups).reshape(len(rows),3,-1)
  p=preference(np.einsum('nvc,v->nc',color_cos,policy['color_views']),engine.scale);best=p.argmax(-1);target=COLORS.index(query['color']);score=p[:,target]
  if query.get('color_negated'):
   valid &= (best!=target)&(p.max(-1)>=policy['color_preference_min']);score=1-score
  else:valid &= (best==target)&(score>=policy['color_preference_min'])
  threshold=policy['color_preference_min'];extra={'color_alternatives':COLORS,'color_prompt_noun':noun}
 else:
  prompt=query['semantic_prompt'];negative=contrasts(prompt,noun)
  groups=[templates(prompt),*[templates(s) for s in negative],templates(category_baseline(prompt,noun))]
  validate_tokens(engine,groups)
  e=np.einsum('nvd,v->nd',emb,np.asarray(policy['description_views']))
  cos=engine.score(e,groups);p=preference(cos[:,:-1],engine.scale);score=p[:,0]
  # Compare the complete description to the category baseline; do not turn a
  # category match alone into proof that extra descriptive conditions hold.
  conditional=1/(1+np.exp(-np.clip((cos[:,0]-cos[:,-1])*engine.scale,-60,60)))
  valid &= (p.argmax(-1)==0)&(score>=policy['description_preference_min'])&(conditional>=policy['query_vs_noun_min'])
  threshold=policy['description_preference_min'];extra={'semantic_prompt':prompt,'counterfactuals':negative}
 objects=[]
 for i,r in enumerate(rows):
  if valid[i]:
   o={k:r[k] for k in ['id','box','detector_score','detector_prompt'] if k in r}
   o.update(semantic_score=float(score[i]),semantic_threshold=threshold,verified_type=names[best_type[i]])
   if query['kind']=='open_color':o['predicted_color']=COLORS[best[i]]
   elif query['kind']=='open_text':o['query_vs_noun_score']=float(conditional[i])
   if guard_scores is not None:o['neutral_type_score']=float(guard_scores[i,guard_best[i]])
   objects.append(o)
 keep=suppress(objects,[o['detector_score']*o['semantic_score'] for o in objects],threshold=policy['nms_iou'])
 selected={objects[i]['id'] for i in keep}
 audit=[]
 for i,row in enumerate(rows):
  order=np.argsort(-type_scores[i])[:4]
  entry={'id':row['id'],'box':row['box'],'selected':row['id'] in selected,'detector_score':row['detector_score'],'detector_prompt':row.get('detector_prompt'),'source_view':row.get('source_view'),
         'type_ranking':[[names[k],float(type_scores[i,k])] for k in order],
         'semantic_score':float(score[i])}
  if query['kind']=='open_color':
   entry['preferred_color']=COLORS[best[i]]
   entry['color_ranking']=[[COLORS[k],float(p[i,k])] for k in np.argsort(-p[i])]
  elif query['kind']=='open_text':entry['query_vs_noun_score']=float(conditional[i])
  if guard_scores is not None:
   entry['category_guard']={'passed':bool(guard_valid[i]),'type':names[guard_best[i]],
                            'score':float(guard_scores[i,guard_best[i]]),'threshold':guard['preference_min'],
                            'view':guard['view'],'color_conditioned':False}
   if inset_scores is not None:
    entry['category_guard'].update(inset_type=names[inset_best[i]],inset_score=float(inset_scores[i,inset_best[i]]))
  audit.append(entry)
 # Prompt embeddings are cheap CPU data, but public inputs must not grow caches forever.
 if len(engine.cache)>128:engine.cache.clear()
 active=crop_trace['active_views']
 raw={'scale':engine.scale,'type_names':names,'computed_view_indices':active,'type_cosines':audit_cosines(type_cos,active)}
 if guard_scores is not None:extra['independent_type_guard']=guard
 if conditioned_type_cos is not None:raw.update(neutral_type_cosines=audit_cosines(neutral_type_cos,active),conditioned_type_cosines=audit_cosines(conditioned_type_cos,active))
 if query['kind']=='open_color':raw.update(color_names=COLORS,color_cosines=audit_cosines(color_cos,active))
 # Optional private diagnostic only. These scores never affect the result.
 # Reconstruct the old car recipe on the SAME noun-pass candidates to explain
 # rejection by the frozen CV threshold, without rerunning/training any model.
 if policy.get('audit_legacy_car') and noun=='car' and query['kind']=='open_color':
  from src.semantic_expert_features import FIXED,COMMON
  legacy_crops=[image.crop(views_for(r['box'],image.size,'vehicle')[1]) for r in rows]
  try:context075=engine.encode(legacy_crops)
  finally:
   for c in legacy_crops:c.close()
  legacy_emb=(emb[:,0]+context075+emb[:,1])/3
  groups=[[f'a photo of a {c} car.',f'a car with {c} bodywork.'] for c in FIXED]
  legacy_p=preference(engine.score(legacy_emb,groups),engine.scale)
  legacy_types=preference(engine.score(emb[:,0],list(TYPE_TEXT.values())),engine.scale)
  selection=json.loads((Path(__file__).resolve().parents[1]/'outputs/a3_color_cv_20261114/selection.json').read_text())
  raw['legacy_car']={'colors':FIXED,'scores':legacy_p.tolist(),'type_names':list(TYPE_TEXT),
                     'type_scores':legacy_types.tolist(),'selected_policy':selection['selected']['policy'],
                     'threshold_color_order':COMMON,'diagnostic_only':True}
 return {'objects':[objects[i] for i in keep],'candidates':len(rows),'policy':policy,'policy_sha256':sha,'prompt_trace':extra,'candidate_audit':audit,'score_audit':raw,'crop_execution':crop_trace}
