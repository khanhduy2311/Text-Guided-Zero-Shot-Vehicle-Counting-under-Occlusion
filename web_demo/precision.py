"""Serving-only FP32 adapter for pre-Ampere hardware; frozen research code unchanged."""
def wrap_fp32(model):
 import torch
 class FullPrecision(torch.nn.Module):
  def __init__(self,base):super().__init__();self.base=base
  def __getattr__(self,name):
   try:return super().__getattr__(name)
   except AttributeError:return getattr(super().__getattr__('base'),name)
  def forward(self,*args,**kwargs):
   # Frozen inference helpers enter BF16 autocast. Disable it only at the
   # actual model boundary on GPUs without native BF16, keeping FP32 weights.
   with torch.autocast('cuda',enabled=False):return self.base(*args,**kwargs)
 return FullPrecision(model)
