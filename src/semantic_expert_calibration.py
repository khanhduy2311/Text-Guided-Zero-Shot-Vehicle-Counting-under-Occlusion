"""Small regularized calibration heads; fitting/selection are split from evaluation."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

def fit_logistic(x,y,c):
 x=np.asarray(x,float);y=np.asarray(y,float);mean=x.mean(0);std=x.std(0);std[std<1e-8]=1.;z=(x-mean)/std
 if len(np.unique(y))<2:
  p=(y.sum()+1)/(len(y)+2);return {'mean':mean.tolist(),'std':std.tolist(),'coef':[0.]*x.shape[1],'intercept':float(np.log(p/(1-p))),'constant':True,'C':c}
 initial=np.zeros(z.shape[1]+1);initial[-1]=np.log((y.sum()+.5)/(len(y)-y.sum()+.5));reg=1/(c*len(y))
 def objective(w):
  logits=z@w[:-1]+w[-1];loss=np.mean(np.logaddexp(0,logits)-y*logits)+reg*np.dot(w[:-1],w[:-1])/2;residual=expit(logits)-y;gradient=np.r_[z.T@residual/len(y)+reg*w[:-1],residual.mean()];return loss,gradient
 r=minimize(objective,initial,jac=True,method='L-BFGS-B',options={'maxiter':400,'ftol':1e-11});assert r.success or r.message.startswith('STOP:'),r.message
 return {'mean':mean.tolist(),'std':std.tolist(),'coef':r.x[:-1].tolist(),'intercept':float(r.x[-1]),'constant':False,'C':c,'optimizer_success':bool(r.success)}
def predict(head,x):
 return expit(((np.asarray(x)-np.asarray(head['mean']))/np.asarray(head['std']))@np.asarray(head['coef'])+head['intercept'])
def binary_metrics(y,decision):
 y=np.asarray(y,bool);p=np.asarray(decision,bool);tp=int((y&p).sum());fp=int((~y&p).sum());fn=int((y&~p).sum());tn=int((~y&~p).sum());return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,'accuracy':float((y==p).mean()) if len(y) else None}
def average_precision(y,score):
 y=np.asarray(y,int);score=np.asarray(score,float)
 if not y.sum():return None
 order=np.argsort(-score,kind='stable');y=y[order];s=score[order];ends=np.r_[np.nonzero(s[1:]!=s[:-1])[0],len(s)-1];tp=y.cumsum()[ends];precision=tp/(ends+1);recall=tp/y.sum();return float(np.dot(np.diff(np.r_[0,recall]),precision))
def choose_threshold(y,p):
 if np.asarray(y).sum()==0:return .5
 candidates=[]
 for threshold in np.arange(.05,1.,.05):
  m=binary_metrics(y,p>=threshold);candidates.append(((m['f1'] or 0,m['precision'] or 0,threshold),threshold))
 return float(max(candidates)[1])
def load_features(folder):
 import json
 from pathlib import Path
 folder=Path(folder);chunks=json.loads((folder/'chunks.json').read_text());ids=[];arrays={}
 for c in chunks:
  with np.load(folder/c['file'],allow_pickle=False) as f:
   ids.extend(f['ids'].tolist())
   for k in f.files:
    if k!='ids':arrays.setdefault(k,[]).append(f[k])
 return ids,{k:np.concatenate(v) for k,v in arrays.items()},json.loads((folder/'metadata.json').read_text())
