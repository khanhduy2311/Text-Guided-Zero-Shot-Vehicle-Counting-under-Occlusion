"""Count, uncertainty interval and detection metrics for the frozen scene audit."""
import numpy as np
from scipy.optimize import linear_sum_assignment

def eligible(box,roi,minimum):
 x1,y1,x2,y2=box;cx=(x1+x2)/2;cy=(y1+y2)/2
 return min(x2-x1,y2-y1)>=minimum and roi[0]<=cx<roi[2] and roi[1]<=cy<roi[3]
def iou_matrix(a,b):
 a=np.asarray(a,dtype=float).reshape(-1,4);b=np.asarray(b,dtype=float).reshape(-1,4)
 lt=np.maximum(a[:,None,:2],b[None,:,:2]);rb=np.minimum(a[:,None,2:],b[None,:,2:]);wh=np.maximum(0,rb-lt);intersection=wh[:,:,0]*wh[:,:,1]
 area_a=np.prod(np.maximum(0,a[:,2:]-a[:,:2]),axis=1);area_b=np.prod(np.maximum(0,b[:,2:]-b[:,:2]),axis=1)
 return intersection/np.maximum(area_a[:,None]+area_b[None,:]-intersection,1e-12)
def match(a,b,threshold=.5):
 m=iou_matrix(a,b)
 if not m.size:return []
 r,c=linear_sum_assignment(-((m>=threshold)*10000+m))
 return [(int(i),int(j),float(m[i,j])) for i,j in zip(r,c) if m[i,j]>=threshold]
def detection(pred,positive,uncertain):
 matched=match(pred,positive);used={i for i,j,v in matched};remaining=[(i,b) for i,b in enumerate(pred) if i not in used]
 ignored=match([b for i,b in remaining],uncertain);ignored_ids={remaining[i][0] for i,j,v in ignored}
 tp=len(matched);fp=len(pred)-tp-len(ignored_ids);fn=len(positive)-tp
 return {'tp':tp,'fp':fp,'fn':fn,'ignored_uncertain':len(ignored_ids),'matches':matched}
def detection_summary(tp,fp,fn):
 return {'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,'tp':tp,'fp':fp,'fn':fn}
def interval_errors(pred,lower,upper):
 assert lower<=upper
 return max(lower-pred,pred-upper,0),max(abs(pred-lower),abs(pred-upper))
def count_summary(pred,gt):
 p=np.asarray(pred,dtype=float);g=np.asarray(gt,dtype=float)
 if not len(g):return {'images':0,'mae':None,'rmse':None,'bias':None,'exact_count_accuracy':None,'wape_percent':None,'positive_images':0,'positive_mae':None,'zero_images':0,'zero_false_positive_rate':None,'gt_total':0,'pred_total':0,'presence_precision':None,'presence_recall':None,'presence_f1':None}
 e=p-g;pos=g>0;zero=~pos;tp=int(((p>0)&pos).sum());fp=int(((p>0)&zero).sum());fn=int(((p==0)&pos).sum())
 return {'images':len(g),'mae':float(abs(e).mean()),'rmse':float(np.sqrt((e*e).mean())),'bias':float(e.mean()),'exact_count_accuracy':float((e==0).mean()),'wape_percent':float(100*abs(e).sum()/g.sum()) if g.sum() else None,'positive_images':int(pos.sum()),'positive_mae':float(abs(e[pos]).mean()) if pos.any() else None,'zero_images':int(zero.sum()),'zero_false_positive_rate':float((p[zero]>0).mean()) if zero.any() else None,'gt_total':int(g.sum()),'pred_total':int(p.sum()),**{'presence_'+k:v for k,v in detection_summary(tp,fp,fn).items() if k in ['precision','recall','f1']}}
