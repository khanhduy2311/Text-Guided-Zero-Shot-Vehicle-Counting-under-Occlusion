"""Trainable heads on frozen SigLIP2 embeddings; shared by evaluation/runtime."""
import numpy as np
import torch
from torch import nn
from src.semantic_expert_features import COMMON,preference

COLORS15=['beige','black','blue','brown','gold','green','grey','orange','pink','purple','red','silver','tan','white','yellow']

def embedding_features(arrays,variant='rich'):
 emb=arrays['embedding'].astype(np.float32)
 if variant=='tight':x=emb[:,0]
 elif variant=='mean':x=emb.mean(1)
 else:x=np.concatenate([emb[:,0],emb[:,1:3].mean(1),emb[:,3]],1)
 # Geometric size is omitted from color learning to avoid crop-domain shortcuts.
 return np.concatenate([x,arrays['chroma'].reshape(len(x),-1),arrays['body_chroma']],1).astype(np.float32)

def make_head(dim,classes,hidden=0):
 return nn.Sequential(nn.Linear(dim,hidden),nn.GELU(),nn.Dropout(.2),nn.Linear(hidden,classes)) if hidden else nn.Linear(dim,classes)

def head_predict(state,arrays):
 x=embedding_features(arrays,state['variant']);z=(x-np.asarray(state['mean']))/np.asarray(state['std']);model=make_head(x.shape[1],state['classes'],state['hidden']);model.load_state_dict(state['weights']);model.eval()
 with torch.inference_mode():
  result=[]
  for start in range(0,len(x),512):result.append(model(torch.as_tensor(z[start:start+512],dtype=torch.float32)).softmax(1).numpy())
 return np.concatenate(result)

def common_colors(p):
 return np.stack([p[:,[i for i,c in enumerate(COLORS15) if c==color or color=='gray' and c in ['grey','silver']]].sum(1) for color in COMMON],1)

def relation_features(arrays):
 emb=arrays['embedding'];prob=preference(arrays['relation'],100.)
 return np.concatenate([emb[:,0],emb[:,1],emb[:,3],prob.reshape(len(emb),-1),arrays['types'][:,0]],1)

def validity_features(arrays):
 return np.concatenate([arrays['embedding'][:,0],arrays['types'][:,0],arrays['geometry']],1)

def pair_geometry(person,vehicle,size):
 px,py,px2,py2=person;vx,vy,vx2,vy2=vehicle;vw=max(vx2-vx,1);vh=max(vy2-vy,1);pw=max(px2-px,1);ph=max(py2-py,1)
 overlap=max(0,min(px2,vx2)-max(px,vx))/min(pw,vw)
 return [(px+px2-vx-vx2)/(2*vw),(py+py2-vy-vy2)/(2*vh),(py2-vy)/vh,pw/vw,ph/vh,overlap,vw/size[0],vh/size[1]]

def binary_feature_head(head,x):
 from src.semantic_expert_calibration import predict
 return predict(head,x)

def pair_features(arrays,rows):
 return np.concatenate([arrays['embedding'][:,0],arrays['pair_relation_cosine'],arrays['pair_types_cosine'],np.asarray([r['pair_geometry'] for r in rows])],1)
