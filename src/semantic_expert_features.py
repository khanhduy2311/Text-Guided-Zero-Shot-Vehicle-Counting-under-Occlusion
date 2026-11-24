"""Shared frozen-VLM feature engine for calibration and deployed inference."""
import math
import numpy as np
import torch
from src.semantic_counting_v2 import crop_box,rider_crop
from src.semantic_chroma import chromatic_ratio
CANONICAL=['black','blue','brown','gray','green','orange','pink','red','tan','violet','white','yellow']
COMMON=['black','blue','gray','green','orange','red','white','yellow']
LEGACY=['green','black','blue','red','white','yellow','gray','silver','orange']
FIXED=['green','black','blue','red','white','yellow','gray','silver','orange','brown']
TYPE_TEXT={'car':['a photo of a car.','a photo of a passenger car or SUV.'],'van':['a photo of a van.','a photo of a passenger minivan.'],'pickup':['a photo of a pickup truck.','a photo of a pickup vehicle.'],'motorcycle':['a photo of a motorcycle.','a photo of a motor scooter.'],'bicycle':['a photo of a bicycle.','a photo of a pedal bicycle.'],'truck':['a photo of a truck.','a photo of a large cargo truck.'],'bus':['a photo of a bus.','a photo of a passenger bus.'],'person':['a photo of a person.','a photo of a pedestrian.'],'background':['a photo of an empty road.','a photo of a building or vegetation.']}
REL_TEXT={'riding':['a photo of a person riding a motorcycle.','a photo of a person seated on a motorcycle.'],'parked':['a photo of a parked motorcycle without a rider.','a photo of an unattended motorcycle.'],'standing':['a photo of a person standing beside a motorcycle.','a photo of a person walking past a motorcycle.']}

def views_for(box,size,domain):
 a=crop_box(box,size);x1,y1,x2,y2=box;w=x2-x1;h=y2-y1
 if domain=='relation':
  b=rider_crop(box,size);c=[max(0,math.floor(x1-.25*w)),max(0,math.floor(y1-h)),min(size[0],math.ceil(x2+.25*w)),min(size[1],math.ceil(y2+.1*h))]
 elif domain=='clothing':b=crop_box(box,size,.075);c=[max(0,math.floor(x1)),max(0,math.floor(y1+.2*h)),min(size[0],math.ceil(x2)),min(size[1],math.ceil(y1+.65*h))]
 else:b=crop_box(box,size,.075);c=[max(0,math.floor(x1+.1*w)),max(0,math.floor(y1+.1*h)),min(size[0],math.ceil(x2-.1*w)),min(size[1],math.ceil(y2-.1*h))]
 return [a,b,c]

def chroma8(image):
 w,h=image.size;rgn=image.crop((int(w*.15),int(h*.15),max(int(w*.15)+1,int(w*.85)),max(int(h*.15)+1,int(h*.85))));rgb=np.asarray(rgn,dtype=np.float32)/255;rgn.close();mx=rgb.max(2);mn=rgb.min(2);delta=mx-mn;safe=np.maximum(delta,1e-6);hue=np.zeros_like(mx);a=rgb.argmax(2)
 for k,offset in [(0,0),(1,2),(2,4)]:
  p=(a==k)&(delta>0);hue[p]=((rgb[:,:,(k+1)%3]-rgb[:,:,(k+2)%3])[p]/safe[p]+offset)%6
 hue*=60;sat=delta/np.maximum(mx,1e-6)
 masks=[mx<.22,(hue>=195)&(hue<270)&(sat>.2)&(mx>.15),(sat<.2)&(mx>=.22)&(mx<=.85),(hue>=70)&(hue<175)&(sat>.25)&(mx>.15),(hue>=15)&(hue<40)&(sat>.3)&(mx>.2),((hue<20)|(hue>=340))&(sat>.25)&(mx>.2),(mx>.7)&(sat<.2),(hue>=35)&(hue<70)&(sat>.3)&(mx>.2)]
 return np.asarray([m.mean() for m in masks],dtype=np.float32)

def preference(cosine,scale):
 z=np.asarray(cosine,dtype=np.float64)*scale;z-=z.max(axis=-1,keepdims=True);p=np.exp(z);return p/p.sum(axis=-1,keepdims=True)

def feature_matrix(arrays,color_index=None,relation=False,scale=100.):
 # Twelve target-specific features, identical for calibration and deployed boxes.
 c=arrays['relation'] if relation else arrays['balanced'];p=preference(c,scale);idx=0 if relation else color_index;other=np.delete(c,idx,axis=-1).max(-1);cols=[]
 for view in range(3):cols.extend([c[:,view,idx],c[:,view,idx]-other[:,view],p[:,view,idx]])
 cols.append(arrays['chroma'][:,0,COMMON.index(CANONICAL[idx])] if not relation else preference(arrays['types'],scale)[:,0,list(TYPE_TEXT).index('motorcycle')]);cols.extend([arrays['geometry'][:,0],arrays['geometry'][:,1]])
 return np.stack(cols,axis=1).astype(np.float64)

class FrozenSemanticFeatures:
 def __init__(self,model,processor,batch_size=8):
  self.model=model;self.processor=processor;self.batch=batch_size;self.cache={};self.scale=float(model.logit_scale.float().exp().item());self.bias=float(model.logit_bias.float().item())
 def text(self,groups):
  key=tuple(tuple(g) for g in groups)
  if key not in self.cache:
   strings=[s for group in groups for s in group]
   with torch.inference_mode():
    tok=self.processor(text=strings,padding='max_length',max_length=64,truncation=True,return_tensors='pt').to('cuda:0');f=self.model.get_text_features(**tok).float();f=(f/f.norm(dim=-1,keepdim=True)).cpu().numpy()
   self.cache[key]=f
  return self.cache[key]
 def score(self,emb,groups):
  values=emb@self.text(groups).T;offset=0;out=[]
  for g in groups:out.append(values[:,offset:offset+len(g)].mean(-1));offset+=len(g)
  return np.stack(out,axis=1)
 def encode(self,crops):
  out=[]
  with torch.inference_mode():
   for i in range(0,len(crops),self.batch):
    pixels=self.processor(images=crops[i:i+self.batch],return_tensors='pt')['pixel_values'].to('cuda:0',dtype=torch.float32);e=self.model.get_image_features(pixel_values=pixels).float();out.append((e/e.norm(dim=-1,keepdim=True)).cpu().numpy())
  return np.concatenate(out)
 def extract(self,image,items):
  boxes=[views_for(r['box'],image.size,r['domain']) for r in items];crops=[image.crop(b) for group in boxes for b in group];emb=self.encode(crops).reshape(len(items),3,-1);output={k:[] for k in ['balanced','native','fixed','types','relation','legacy_relation','chroma','body_ratio','geometry']}
  for i,r in enumerate(items):
   noun=r['noun'];clothes=r['domain']=='clothing'
   def phrase(c,native=False):
    if clothes:return [f'a photo of a person wearing {c} clothes.'] if native else [f'a photo of a person wearing {c} clothes.',f'a person in {c} clothing.']
    return [f'a photo of a {c} {noun}.'] if native else [f'a photo of a {c} {noun}.',f'a {noun} with {c} bodywork.']
   canon=[phrase('gray or silver' if c=='gray' else ('violet or purple' if c=='violet' else c)) for c in CANONICAL]
   output['balanced'].append(self.score(emb[i],canon));output['native'].append(self.score(emb[i],[phrase(c,True) for c in LEGACY]));output['fixed'].append(self.score(emb[i],[phrase(c) for c in FIXED]));output['types'].append(self.score(emb[i],list(TYPE_TEXT.values())));output['relation'].append(self.score(emb[i],list(REL_TEXT.values())));output['legacy_relation'].append(self.score(emb[i],[['a photo of person riding a motorcycle'],['a photo of an ordinary regular vehicle'],['a photo of an empty street road']]))
   cur=crops[i*3:i*3+3];output['chroma'].append(np.stack([chroma8(im) for im in cur]));output['body_ratio'].append(np.asarray([[chromatic_ratio(im,c) for c in ['red','blue']] for im in cur]));b=r['box'];output['geometry'].append([math.log1p((b[2]-b[0])*(b[3]-b[1])),math.log((b[2]-b[0])/(b[3]-b[1]))])
  for im in crops:im.close()
  return {k:np.asarray(v,dtype=np.float32) for k,v in output.items()}
