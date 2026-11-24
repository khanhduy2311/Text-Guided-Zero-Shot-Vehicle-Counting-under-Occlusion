"""Small nonnegative color-score fusion, fitted with image-group cross-validation."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit
from src.full_semantic_policy import select_boxes
from src.a3_semantic_refinement import metrics

def color_inputs(scores):
 return np.stack([logit(np.clip(scores[k],1e-6,1-1e-6)) for k in ['trained_color','multiview_color','body_color']],2)

def fit_color(x,y,weights,groups,c):
 known=y>=0;x=x[known];y=y[known];weights=weights[known];groups=[groups[i] for i in np.flatnonzero(known)]
 positives=len({g for g,t in zip(groups,y) if t==1})
 if positives<3:return {'fallback':True,'positive_groups':positives,'C':c}
 prior=np.array([.5,.5,0.,0.]);penalty=np.array([1.,1.,1.,.1]);reg=1/(c*weights.sum())
 def objective(w):
  z=x@w[:3]+w[3];res=(expit(z)-y)*weights/weights.sum();diff=w-prior
  loss=np.dot(weights,np.logaddexp(0,z)-y*z)/weights.sum()+reg*np.dot(penalty*diff,diff)/2
  grad=np.r_[x.T@res,res.sum()]+reg*penalty*diff
  return loss,grad
 result=minimize(objective,prior,jac=True,method='L-BFGS-B',bounds=[(0,None)]*3+[(None,None)],options={'maxiter':500,'ftol':1e-12})
 assert result.success,result.message
 return {'fallback':False,'coef':result.x[:3].tolist(),'intercept':float(result.x[3]),'positive_groups':positives,'C':c,'optimizer_success':bool(result.success)}

def apply_heads(x,heads):
 output=[]
 for j,head in enumerate(heads):
  if head['fallback']:output.append(.5*expit(x[:,j,0])+.5*expit(x[:,j,1]))
  else:output.append(expit(x[:,j]@np.asarray(head['coef'])+head['intercept']))
 return np.stack(output,1)

def score_mode(scores,heads,mode):
 if mode=='multiview':return scores['multiview_color']
 if mode=='vcor':return scores['trained_color']
 if mode=='blend':return .5*scores['trained_color']+.5*scores['multiview_color']
 return apply_heads(color_inputs(scores),heads[mode])

def predict(rows,scores,selection):
 policy=selection['selected']['policy'];ix=select_boxes(rows,scores,policy['boxes']);p=score_mode(scores,selection['heads'],policy['mode'])[ix];d=p>=np.asarray(policy['threshold']);return [rows[i] for i in ix],p,d,ix

def evaluate(rows,scores,selection,images,gt):
 rr,p,d,ix=predict(rows,scores,selection);return metrics(rr,d,images,gt,{},'vehicle',images[0]['role'])
