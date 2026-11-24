"""Versioned semantic routing: vehicle-first relation crops and balanced contrasts."""
import math

def crop_box(box,size,pad=.0):
 x1,y1,x2,y2=box;w=x2-x1;h=y2-y1;return [max(0,math.floor(x1-pad*w)),max(0,math.floor(y1-pad*h)),min(size[0],math.ceil(x2+pad*w)),min(size[1],math.ceil(y2+pad*h))]
def rider_crop(box,size):
 x1,y1,x2,y2=box;w=x2-x1;h=y2-y1;headroom=.1 if h/max(w,1)>1.3 else .5
 return [max(0,math.floor(x1-.08*w)),max(0,math.floor(y1-headroom*h)),min(size[0],math.ceil(x2+.08*w)),min(size[1],math.ceil(y2+.05*h))]
def rider_upper_crop(box,size):
 x1,y1,x2,y2=box;h=y2-y1;w=x2-x1;r=rider_crop(box,size);r[3]=min(size[1],math.ceil(y1+(.65 if h/max(w,1)>1.3 else .2)*h));return r

def iou(a,b):
 x=max(0,min(a[2],b[2])-max(a[0],b[0]));y=max(0,min(a[3],b[3])-max(a[1],b[1]));z=x*y;aa=(a[2]-a[0])*(a[3]-a[1]);bb=(b[2]-b[0])*(b[3]-b[1]);return z/max(aa+bb-z,1e-12)
def deduplicate(objects):
 """Remove near-identical boxes only; ordinary crowded overlapping objects survive."""
 kept=[];removed=[]
 for o in sorted(objects,key=lambda o:(-o['detector_score'],o['id'])):
  b=o['box'];same=None
  for k in kept:
   a=k['box'];distance=math.hypot((a[0]+a[2]-b[0]-b[2])/2,(a[1]+a[3]-b[1]-b[3])/2);short=min(a[2]-a[0],a[3]-a[1],b[2]-b[0],b[3]-b[1])
   if iou(a,b)>=.85 and distance<=.1*short:same=k['id'];break
  if same is None:kept.append(o)
  else:removed.append({'id':o['id'],'duplicate_of':same})
 return kept,removed

def ranking_state(labels,probabilities,target,minimum):
 if len(labels)!=len(probabilities) or not probabilities or any(not math.isfinite(p) for p in probabilities):raise ValueError('Finite aligned preference values required')
 index=max(range(len(labels)),key=lambda i:probabilities[i]);label=labels[index];top=probabilities[index]
 if label!=target:return 'rejected'
 return 'matched' if top>=minimum else 'unknown'

def type_matches(label,target):
 if target=='vehicle':return label in ['car','van','pickup','motorcycle','bicycle','bus','truck']
 if target=='car':return label in ['car','van','pickup']
 return label==target
