"""Frozen prediction policies shared by held-out evaluation and deployment."""
import numpy as np
from scipy.special import expit
from src.semantic_expert_features import COMMON,CANONICAL,LEGACY,FIXED,preference,feature_matrix
from src.semantic_expert_calibration import predict

def color_predictions(arrays,metadata,domain,expert):
 result={};n=len(arrays['geometry']);scale=metadata['scale']
 for name,key,colors in [('legacy_rule','native',LEGACY),('fixed_rule','fixed',FIXED)]:
  p=preference(arrays[key][:,0],scale);decision=np.zeros((n,len(COMMON)),bool);score=np.zeros((n,len(COMMON)))
  top=p.argmax(1)
  for j,color in enumerate(COMMON):
   indices=[i for i,c in enumerate(colors) if c==color or (color=='gray' and c=='silver')];score[:,j]=p[:,indices].sum(1);decision[:,j]=np.isin(top,indices)&(p.max(1)>=.25)
   if name=='fixed_rule' and domain=='vehicle' and color in ['red','blue']:decision[:,j]&=arrays['body_ratio'][:,0,['red','blue'].index(color)]>=({'red':.07,'blue':.06}[color])
  result[name]=(score,decision)
 score=np.stack([predict(expert['domains'][domain]['heads'][c],feature_matrix(arrays,CANONICAL.index(c),scale=scale)) for c in COMMON],1)
 threshold=np.asarray([expert['domains'][domain]['heads'][c]['threshold'] for c in COMMON]);result['calibrated']=(score,score>=threshold)
 return result

def relation_predictions(arrays,metadata,expert=None):
 scale=metadata['scale'];p=preference(arrays['legacy_relation'][:,0],scale);absolute=expit(arrays['legacy_relation'][:,0,0]*scale+metadata['bias']);result={'legacy_rule':(p[:,0],(absolute>=.15)&(p[:,0]>=.4))}
 p=preference(arrays['relation'][:,1],scale);result['fixed_rule']=(p[:,0],(p.argmax(1)==0)&(p[:,0]>=.4))
 if expert is not None:
  score=predict(expert['head'],feature_matrix(arrays,relation=True,scale=scale));result['calibrated']=(score,score>=expert['head']['threshold'])
 return result
