"""Optional body-color safeguard, matching existing Cubi AttributeVerifier defaults."""
import copy
from pathlib import Path
import numpy as np
from PIL import Image

def chromatic_ratio(crop,color):
 w,h=crop.size;region=crop.crop((int(w*.15),int(h*.15),max(int(w*.15)+1,int(w*.85)),max(int(h*.15)+1,int(h*.85))));a=np.asarray(region,dtype=np.float32);region.close();r,g,b=a[...,0],a[...,1],a[...,2]
 if color=='red':mask=(r>60)&(r>1.25*g)&(r>1.25*b)&((r-np.maximum(g,b))>20)
 elif color=='blue':mask=(b>60)&(b>1.25*r)&(b>1.25*g)&((b-np.maximum(r,g))>20)
 else:return 1.
 return float(mask.mean())

def apply_body_chroma(result,root,strict_car=False):
 result=copy.deepcopy(result)
 for im in result['images']:
  with Image.open(Path(root)/im['image']) as image:
   image=image.convert('RGB')
   for q in im['queries']:
    color=q['query'].get('target_color')
    if strict_car and q['kind']=='attribute' and q['query'].get('target_type')=='car':
     for obj in q['objects']:
      if obj['status']=='matched' and obj.get('type_verification',{}).get('label')!='car':obj.update(status='rejected',reason='target_car_excludes_van_pickup')
    if q['kind']!='attribute' or color not in ['red','blue']:continue
    for obj in q['objects']:
     obj['status_before_chroma']=obj['status']
     if obj['status']!='matched':continue
     crop=image.crop(obj['crop_box']);ratio=chromatic_ratio(crop,color);crop.close();minimum=.07 if color=='red' else .06;obj['body_chromatic_ratio']=ratio;obj['body_chromatic_minimum']=minimum
     if ratio<minimum:obj.update(status='rejected',reason='insufficient_body_color_pixels')
    for status,key in [('matched','count'),('unknown','unknown_count'),('rejected','rejected_count')]:q[key]=sum(o['status']==status for o in q['objects'])
 result['body_chroma_policy']={'source':'Cubi/script/attribute_verifier.py existing compute_chromatic_ratio and thresholds','red_minimum':.07,'blue_minimum':.06,'central_fraction':.7,'strict_car_type':strict_car,'applied_to':'car-body attributes only; never rider clothing','note':'Heuristic paint safeguard, can reject dark/desaturated true targets.'};result['engine']='balanced_contrasts_vehicle_centered_v2_body_chroma'
 return result
