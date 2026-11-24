"""A3/A4/A5: encoder feature consistency and optional mask-free inference refinement.

The hook returns corrected memory to the native transformer BEFORE proposal
selection and decoder. Ground-truth/synthetic masks are loss arguments only.
"""
import torch
from torch import nn
import torch.nn.functional as F


class ContextRefiner(nn.Module):
    def __init__(self, channels=256, hidden=64, predicted_gate=False, fixed_gate=.5):
        super().__init__()
        self.predicted_gate=predicted_gate; self.fixed_gate=fixed_gate
        self.project=nn.Conv2d(channels,hidden,1)
        self.context=nn.ModuleList([nn.Conv2d(hidden,hidden,3,padding=d,dilation=d,groups=hidden) for d in (1,2,4,8)])
        self.query_norm=nn.LayerNorm(hidden)
        self.text_project=nn.Linear(channels,hidden)
        self.attention=nn.MultiheadAttention(hidden,4,batch_first=True,dropout=0.)
        self.output=nn.Conv2d(hidden,channels,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.gate=nn.Conv2d(hidden,1,1) if predicted_gate else None
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight);nn.init.constant_(self.gate.bias,-2.)

    def forward(self,memory,text,shapes,padding,text_padding):
        batch,_,channels=memory.shape; sizes=[tuple(map(int,s)) for s in shapes.tolist()]
        chunks=memory.split([h*w for h,w in sizes],dim=1)
        masks=padding.split([h*w for h,w in sizes],dim=1)
        maps=[x.transpose(1,2).reshape(batch,channels,h,w) for x,(h,w) in zip(chunks,sizes)]
        hidden=[self.project(x) for x in maps]
        pooled=[]
        for x,mask,(h,w) in zip(hidden,masks,sizes):
            valid=(~mask).reshape(batch,1,h,w).to(x.dtype)
            pooled.append((x*valid).sum((2,3),keepdim=True)/valid.sum((2,3),keepdim=True).clamp_min(1))
        cross_level=torch.stack(pooled).mean(0)
        tokens=self.text_project(text)
        result=[];logits=[]
        for original,x,mask,(h,w) in zip(maps,hidden,masks,sizes):
            x=F.gelu(x+sum(layer(x) for layer in self.context)/len(self.context)+cross_level)
            q=self.query_norm(x.flatten(2).transpose(1,2))
            contextual,_=self.attention(q,tokens,tokens,key_padding_mask=text_padding,need_weights=False)
            x=x+contextual.transpose(1,2).reshape_as(x)
            delta=self.output(F.gelu(x))
            gate_logits=self.gate(x) if self.gate is not None else None
            gate=gate_logits.sigmoid() if gate_logits is not None else self.fixed_gate
            valid=(~mask).reshape(batch,1,h,w)
            corrected=original+delta*gate*valid
            result.append(corrected.flatten(2).transpose(1,2))
            logits.append(gate_logits)
        return torch.cat(result,dim=1),logits


class EncoderTap:
    def __init__(self,model,refiner=None):
        self.refiner=refiner;self.capture=False;self.last=None;self.decoder_seen=False
        self.handle=model.transformer.encoder.register_forward_hook(self.hook,with_kwargs=True)
        self.decoder_handle=model.transformer.decoder.register_forward_pre_hook(self.decoder_hook,with_kwargs=True)

    def hook(self,module,args,kwargs,output):
        memory,text=output;shapes=kwargs['spatial_shapes'];padding=kwargs['key_padding_mask']
        gates=None
        if self.refiner is not None:
            memory,gates=self.refiner(memory,text,shapes,padding,kwargs['text_attention_mask'])
        if self.capture:
            self.last={'memory':memory,'shapes':shapes,'padding':padding,'gates':gates}
        return memory,text

    def decoder_hook(self,module,args,kwargs):
        if self.capture:
            if self.last is None or kwargs['memory'].data_ptr()!=self.last['memory'].data_ptr():
                raise RuntimeError('Decoder did not receive the encoder/refiner memory')
            self.decoder_seen=True

    def clear(self):self.last=None;self.decoder_seen=False
    def close(self):self.handle.remove();self.decoder_handle.remove();self.clear()


class _EncoderResult(Exception):
    def __init__(self,value):self.value=value


class FrozenTeacher:
    """Stop clean no-grad teacher after encoder; no unnecessary query decoding."""
    def __init__(self,model,nested_factory,device):
        self.model=model;self.nested_factory=nested_factory;self.device=device
        model.eval().requires_grad_(False)
        self.handle=model.transformer.encoder.register_forward_hook(self.hook,with_kwargs=True)

    @staticmethod
    def hook(module,args,kwargs,output):
        raise _EncoderResult({'memory':output[0].detach(),'shapes':kwargs['spatial_shapes'],
            'padding':kwargs['key_padding_mask']})

    @torch.no_grad()
    def features(self,images,captions):
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError('Teacher must remain frozen and eval')
        samples=self.nested_factory([im.to(self.device) for im in images])
        empty=[torch.empty(0,4,device=self.device) for _ in images]
        labels=[torch.empty(0,dtype=torch.long,device=self.device) for _ in images]
        try:
            with torch.autocast('cuda',dtype=torch.bfloat16):self.model(samples,empty,labels,captions=captions)
        except _EncoderResult as result:return result.value
        raise RuntimeError('Teacher encoder hook did not execute')

    def close(self):self.handle.remove()


def spatial_objectives(student,teacher,occlusion_masks,config):
    """Per-image, per-level normalized loss, excluding image padding."""
    if not torch.equal(student['shapes'],teacher['shapes']) or not torch.equal(student['padding'],teacher['padding']):
        raise ValueError('Clean/occluded geometry is not aligned')
    s=student['memory'];t=teacher['memory'].detach()
    if s.shape!=t.shape:raise ValueError('Teacher/student feature shapes differ')
    batch=s.shape[0];device=s.device
    padded_h=max(m.shape[0] for m in occlusion_masks);padded_w=max(m.shape[1] for m in occlusion_masks)
    masks=torch.zeros(batch,1,padded_h,padded_w,device=device)
    for i,m in enumerate(occlusion_masks):masks[i,0,:m.shape[0],:m.shape[1]]=m.to(device)
    feature=s.new_zeros(batch,dtype=torch.float32);gate_loss=feature.clone();start=0
    shapes=student['shapes'].tolist()
    for level,(h,w) in enumerate(shapes):
        n=h*w;valid=(~student['padding'][:,start:start+n]).float()
        added=F.interpolate(masks,size=(h,w),mode='area').flatten(1)
        x=F.layer_norm(s[:,start:start+n].float(),(s.shape[-1],))
        y=F.layer_norm(t[:,start:start+n].float(),(t.shape[-1],))
        distance=1-F.cosine_similarity(x,y,dim=-1)
        distance=distance+config['charbonnier_weight']*((x-y).square()+1e-6).sqrt().mean(-1)
        weights=valid*(1+config['occluded_weight']*added)
        feature+=(distance*weights).sum(-1)/weights.sum(-1).clamp_min(1)
        gates=student.get('gates')
        if gates is not None and gates[level] is not None:
            logits=gates[level].float().flatten(1)
            loss=F.binary_cross_entropy_with_logits(logits,added,reduction='none',pos_weight=logits.new_tensor(config['gate_positive_weight']))
            gate_loss+=(loss*valid).sum(-1)/valid.sum(-1).clamp_min(1)
        start+=n
    feature/=len(shapes);gate_loss/=len(shapes)
    if not torch.isfinite(feature).all() or not torch.isfinite(gate_loss).all():raise ValueError('Nonfinite spatial loss')
    return feature,gate_loss
