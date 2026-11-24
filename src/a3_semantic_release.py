"""Versioned localization/color ablation without changing phase1 frozen code."""
import numpy as np
from src.semantic_expert_features import COMMON,FIXED,preference
from src.a3_semantic_refinement import infer_scores,gate_mask
def release_predictions(arrays,rows,metadata,policy,heads):
 if policy['kind']!='fixed':return infer_scores(arrays,rows,metadata,policy,heads)
 p=preference(arrays['fixed'][:,0],metadata['scale']);scores=[];decisions=[];top=p.argmax(1)
 for color in COMMON:
  indices=[i for i,c in enumerate(FIXED) if c==color or color=='gray' and c=='silver'];scores.append(p[:,indices].sum(1));d=np.isin(top,indices)&(p.max(1)>=.25)
  if color in ['red','blue']:d&=arrays['body_ratio'][:,0,['red','blue'].index(color)]>=({'red':.07,'blue':.06}[color])
  decisions.append(d)
 mask=gate_mask(arrays,'vehicle',policy['gate'],heads,rows,metadata);return np.stack(scores,1),np.stack(decisions,1)&mask[:,None]
