"""Encode only used, distinct pixel crops; no proposal or decision is removed."""
def active_views(policy,query):
 weights=[[1,0,0] if query['kind']=='count' else policy['type_views']]
 if query['kind']!='count':weights.append(policy['color_views'] if query['kind']=='open_color' else policy['description_views'])
 guard=policy.get('independent_type_guard',{})
 if query['noun'] in guard.get('nouns',[]):weights.append([1,1 if guard.get('view')=='tight_and_inset' else 0,0])
 if policy.get('audit_legacy_car') and query['noun']=='car' and query['kind']=='open_color':return [0,1,2]
 return [i for i in range(3) if any(w[i]!=0 for w in weights)]

def encode_views(engine,image,boxes,active,optimize):
 import numpy as np
 active=active if optimize else [0,1,2]
 crops=[];lookup={};slots=[]
 for i,views in enumerate(boxes):
  for v in active:
   key=tuple(views[v])
   if not optimize or key not in lookup:
    index=len(crops);crops.append(image.crop(views[v]));lookup[key]=index
   else:index=lookup[key]
   slots.append((i,v,index))
 try:encoded=engine.encode(crops)
 finally:
  for crop in crops:crop.close()
 embedding=np.zeros((len(boxes),3,encoded.shape[-1]),dtype=encoded.dtype)
 for i,v,index in slots:embedding[i,v]=encoded[index]
 trace={'active_views':active,'requested_crops':len(boxes)*3,'encoded_crops':len(crops),
        'unused_views_skipped':len(boxes)*(3-len(active)),'duplicate_crops_reused':len(slots)-len(crops)}
 return embedding,trace

def audit_cosines(array,active):
 # Missing diagnostic views are explicitly null, never fabricated zero scores.
 return [[v.tolist() if i in active else None for i,v in enumerate(row)] for row in array]
