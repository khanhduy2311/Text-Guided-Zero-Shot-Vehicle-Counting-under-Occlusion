"""A3 single-pass proposal recovery with explicit full-image coordinates."""
import math
import numpy as np
import torch
from src.step7_inference import normalize_image,project_candidates
from src.countgdpp_inference import decode_outputs

def overlapping_quadrants(size):
 w,h=size;cw,ch=math.ceil(w*.6),math.ceil(h*.6)
 return [(x,y,x+cw,y+ch) for y in sorted({0,h-ch}) for x in sorted({0,w-cw})]

@torch.inference_mode()
def single(model,image,noun,nested_factory,side=800,maxsize=1333):
 tensor=normalize_image(image,side,maxsize).to('cuda:0');caption=noun+' .';tokens=model.tokenizer(caption)['input_ids']
 with torch.autocast('cuda',dtype=torch.bfloat16):
  output=model(nested_factory([tensor]),[torch.empty((0,4),device='cuda:0')],[torch.empty((0,),device='cuda:0',dtype=torch.long)],captions=[caption])
 boxes,scores,text=decode_outputs(output,tokens,0)
 return {k:v.detach().cpu().numpy() for k,v in [('boxes',boxes),('scores',scores),('text_min',text)]}

def proposal_views(model,image,noun,nested_factory,include_tiles):
 full=(0,0,*image.size);views=[('full800',full,800,1333)]
 if include_tiles:views += [(f'tile{i}',b,640,1024) for i,b in enumerate(overlapping_quadrants(image.size))]
 result=[]
 for name,box,side,maxsize in views:
  crop=image.crop(box);raw=single(model,crop,noun,nested_factory,side,maxsize);crop.close();p=project_candidates(raw,box,image.size,.10,0,0 if name=='full800' else 2)
  # A fixed per-view cap bounds background-heavy scenes before semantic scoring.
  candidates=sorted(zip(p['boxes_xyxy'],p['scores'],p['query_indices']),key=lambda row:(-row[1],row[2]))[:100]
  for b,s,q in candidates:
   if min(b[2]-b[0],b[3]-b[1])>=12:result.append({'box':b,'detector_score':float(s),'source_view':name,'query_index':int(q)})
 return result

def suppress(objects,quality,threshold=.65,containment=.95):
 """Stable NMS; containment only for near-equal areas to retain nested instances."""
 from src.semantic_counting_v2 import iou
 order=sorted(range(len(objects)),key=lambda i:(-float(quality[i]),objects[i]['id']));keep=[]
 for i in order:
  a=objects[i]['box'];area=max(0,a[2]-a[0])*max(0,a[3]-a[1]);duplicate=False
  for j in keep:
   b=objects[j]['box'];other=max(0,b[2]-b[0])*max(0,b[3]-b[1]);inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
   if iou(a,b)>=threshold or (min(area,other)>0 and max(area,other)/min(area,other)<1.5 and inter/min(area,other)>=containment):duplicate=True;break
  if not duplicate:keep.append(i)
 return keep
