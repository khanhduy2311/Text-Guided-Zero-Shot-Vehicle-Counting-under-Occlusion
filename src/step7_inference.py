"""Annotation-free pseudo selection and tiled inference; all merges in original pixels."""
import math,time
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from .countgdpp_inference import decode_outputs,select_pseudo_exemplars
from .counting_eval import box_iou

MODES=('single','pseudo3','filtered_pseudo3','crop_single','filtered_pseudo3_crop')

def tile_boxes(width,height,tile=800,overlap=.25):
    if tile<=0 or not 0<=overlap<1:raise ValueError('Invalid tile geometry')
    if width<=tile and height<=tile:return []
    def starts(n):
        if n<=tile:return [0]
        step=max(1,round(tile*(1-overlap)))
        return sorted(set([*range(0,n-tile+1,step),n-tile]))
    return [(x,y,min(x+tile,width),min(y+tile,height)) for y in starts(height) for x in starts(width)]


def choose_filtered(boxes,scores,size_hw,policy):
    h,w=size_hw
    xyxy=np.concatenate((boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2),axis=-1)*[w,h,w,h]
    widths=xyxy[:,2]-xyxy[:,0];heights=xyxy[:,3]-xyxy[:,1];margin=policy['border_margin_px']
    valid=(scores>policy['threshold'])&(widths>=policy['minimum_side_px'])&(heights>=policy['minimum_side_px'])
    valid&=(xyxy[:,0]>=margin)&(xyxy[:,1]>=margin)&(xyxy[:,2]<=w-margin)&(xyxy[:,3]<=h-margin)
    candidates=np.flatnonzero(valid).tolist();candidates=sorted(candidates,key=lambda i:(-float(scores[i]),i))[:policy['candidate_limit']]
    selected=[]
    while candidates and len(selected)<policy['count']:
        if not selected:index=candidates[0]
        else:
            def quality(i):
                distance=min(float(np.linalg.norm(boxes[i,:2]-boxes[j,:2])) for j in selected)
                return float(scores[i])*(.5+.5*min(1.,distance)), -i
            index=max(candidates,key=quality)
        selected.append(index)
        candidates=[i for i in candidates if i!=index and all(box_iou(xyxy[i:i+1],xyxy[j:j+1])[0,0]<policy['duplicate_iou'] for j in selected)]
    return xyxy[selected].astype(np.float32).reshape(-1,4),selected


def project_candidates(prediction,crop,full_size,threshold,text_threshold,border_margin=0):
    """No GT/ROI filtering; suppress internal crop-edge truncations only."""
    width,height=full_size;x0,y0,x1,y1=crop;cw,ch=x1-x0,y1-y0
    boxes=np.asarray(prediction['boxes']);scores=np.asarray(prediction['scores']);text=np.asarray(prediction['text_min'])
    xyxy=np.concatenate((boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2),axis=-1)*[cw,ch,cw,ch]+[x0,y0,x0,y0]
    xyxy[:,0::2]=np.clip(xyxy[:,0::2],x0,x1);xyxy[:,1::2]=np.clip(xyxy[:,1::2],y0,y1)
    confident=(scores>threshold)&(text>text_threshold)
    edge=np.zeros(len(boxes),dtype=bool)
    if border_margin:
        if x0>0:edge|=xyxy[:,0]<=x0+border_margin
        if y0>0:edge|=xyxy[:,1]<=y0+border_margin
        if x1<width:edge|=xyxy[:,2]>=x1-border_margin
        if y1<height:edge|=xyxy[:,3]>=y1-border_margin
    valid=confident&~edge&(xyxy[:,2]>xyxy[:,0])&(xyxy[:,3]>xyxy[:,1])
    indices=np.flatnonzero(valid)
    return {'boxes_xyxy':xyxy[valid].tolist(),'scores':scores[valid].tolist(),'text_min':text[valid].tolist(),
        'query_indices':indices.tolist(),'edge_discarded_query_indices':np.flatnonzero(confident&edge).tolist(),
        'pre_edge_candidates':{'boxes_xyxy':xyxy[confident].tolist(),'scores':scores[confident].tolist(),'query_indices':np.flatnonzero(confident).tolist()}}


def merge_views(groups,full_size,policy):
    """One-to-one matching across views, never merge detections from the same view."""
    clusters=[]
    for view,group in groups:
        boxes=np.asarray(group['boxes_xyxy'],dtype=float).reshape(-1,4)
        incoming=[{'box':box.copy(),'score':float(score),'text_min':float(text),'members':[(view,int(q))]}
            for box,score,text,q in zip(boxes,group['scores'],group['text_min'],group['query_indices'])]
        if not incoming:continue
        matched=set()
        if clusters:
            old=np.asarray([c['box'] for c in clusters]);iou=box_iou(old,boxes)
            a=np.maximum(old[:,2:]-old[:,:2],1e-6);b=np.maximum(boxes[:,2:]-boxes[:,:2],1e-6)
            delta=(old[:,None,:2]+old[:,None,2:]-boxes[None,:,:2]-boxes[None,:,2:])/2
            center=np.linalg.norm(delta/np.minimum(a[:,None,:],b[None,:,:]),axis=-1)
            ratio=np.maximum(a[:,None,:]/b[None,:,:],b[None,:,:]/a[:,None,:]).max(-1)
            valid=(iou>=policy['iou'])&(center<=policy['center_fraction'])&(ratio<=policy['max_size_ratio'])
            # One extra valid pair outweighs quality, preserving crowded instances.
            cost=np.where(valid,-1-iou/(min(iou.shape)+1),0.)
            rows,cols=linear_sum_assignment(cost)
            for row,col in zip(rows,cols):
                if not valid[row,col]:continue
                cluster=clusters[row];candidate=incoming[col]
                if any(v==view for v,q in cluster['members']):raise ValueError('Duplicate view in merge cluster')
                cluster['members'].extend(candidate['members'])
                if candidate['score']>cluster['score']:
                    cluster.update({k:candidate[k] for k in ('box','score','text_min')})
                matched.add(col)
        clusters.extend(candidate for i,candidate in enumerate(incoming) if i not in matched)
    width,height=full_size
    xyxy=np.asarray([c['box'] for c in clusters],dtype=float).reshape(-1,4)
    normalized=np.concatenate(((xyxy[:,:2]+xyxy[:,2:])/2,xyxy[:,2:]-xyxy[:,:2]),axis=-1)/[width,height,width,height]
    return {'boxes':normalized.astype(np.float32),'scores':np.asarray([c['score'] for c in clusters],dtype=np.float32),
        'text_min':np.asarray([c['text_min'] for c in clusters],dtype=np.float32)},[c['members'] for c in clusters]


def normalize_image(image,short_side,max_size):
    width,height=image.size;scale=min(short_side/min(width,height),max_size/max(width,height))
    size=(max(1,round(width*scale)),max(1,round(height*scale)))
    image=image.resize(size,Image.Resampling.BILINEAR)
    tensor=torch.from_numpy(np.array(image,copy=True)).permute(2,0,1).float()/255
    return (tensor-tensor.new_tensor([.485,.456,.406])[:,None,None])/tensor.new_tensor([.229,.224,.225])[:,None,None]


@torch.inference_mode()
def infer_step7(model,image,prompt,nested_factory,device,config):
    if model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('Frozen eval model required')
    if image.mode!='RGB':image=image.convert('RGB')
    query=' '.join(prompt.replace('_',' ').strip().rstrip('.').split())
    if not query or '.' in query or '?' in query:raise ValueError('Expected one positive query')
    caption=query+' .';ids=model.tokenizer(caption)['input_ids']
    if len(ids)+3>model.max_text_len:raise ValueError('Prompt exceeds model capacity')
    timings=[];selection=[];view_groups={};raw={};started=time.perf_counter()
    def predict(tensor,exemplars,view,kind):
        samples=nested_factory([tensor.to(device)]);exemplars=torch.as_tensor(exemplars,device=device,dtype=torch.float32).reshape(-1,4)
        torch.cuda.synchronize(device);start=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output=model(samples,[exemplars],[torch.zeros(len(exemplars),device=device,dtype=torch.long)],captions=[caption])
        torch.cuda.synchronize(device)
        boxes,scores,text=decode_outputs(output,ids,len(exemplars))
        timings.append({'view':view,'kind':kind,'seconds':time.perf_counter()-start})
        return {k:v.detach().cpu().numpy() for k,v in [('boxes',boxes),('scores',scores),('text_min',text)]}
    def view(image,crop,view_id,full=False):
        tensor=normalize_image(image,config['short_side'] if full else config['crop_short_side'],config['max_size'] if full else config['crop_max_size'])
        first=predict(tensor,[],view_id,'single')
        boxes,chosen=choose_filtered(first['boxes'],first['scores'],tensor.shape[-2:],config['pseudo_filter'])
        filtered=predict(tensor,boxes,view_id,'filtered_pseudo3') if len(boxes) else first
        selection.append({'view':view_id,'query_indices':chosen,'fallback_single':not bool(len(boxes))})
        outputs={'single':first,'filtered_pseudo3':filtered}
        if full:
            pseudo,_=select_pseudo_exemplars(torch.from_numpy(first['boxes']),torch.from_numpy(first['scores']),tensor.shape[-2:],config['box_threshold'],3)
            outputs['pseudo3']=predict(tensor,pseudo,view_id,'pseudo3')
        groups={mode:project_candidates(pred,crop,image_full_size,config['box_threshold'],config['text_threshold'],0 if full else config['crop_border_margin_px']) for mode,pred in outputs.items()}
        raw[view_id]={'crop_xyxy':list(crop),'predictions':groups};view_groups[view_id]=groups
        return outputs
    image_full_size=image.size;width,height=image_full_size
    full=view(image,(0,0,width,height),'full',True)
    crops=tile_boxes(width,height,config['tile_size'],config['tile_overlap'])
    for index,crop in enumerate(crops):view(image.crop(crop),crop,f'crop_{index:03d}')
    result=dict(full);traces={}
    for mode,key in [('crop_single','single'),('filtered_pseudo3_crop','filtered_pseudo3')]:
        if not crops:result[mode]=full[key];traces[mode]=[]
        else:result[mode],traces[mode]=merge_views([(name,groups[key]) for name,groups in view_groups.items()],image_full_size,config['merge'])
    return result,{'views':raw,'selected_exemplars':selection,'merge_members':traces,'forward_timings':timings,
        'wall_seconds':time.perf_counter()-started,'crop_count':len(crops),'original_size_wh':[width,height]}
