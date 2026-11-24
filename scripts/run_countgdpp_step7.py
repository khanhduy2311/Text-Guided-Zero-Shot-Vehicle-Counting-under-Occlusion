"""Frozen source-only step-7 inference ablations; never trains or loads target images."""
import argparse,fcntl,gc,hashlib,inspect,json,math,os,socket,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'CountGDPlusPlus'))
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import numpy as np
import torch
from PIL import Image
from util.slconfig import SLConfig
from groundingdino.util.misc import nested_tensor_from_tensor_list
from models.GroundingDINO.groundingdino import build_groundingdino
from models.GroundingDINO import ms_deform_attn
from src.counting_data.loader import UnifiedCountingDataset
from src.counting_eval import evaluate_image,summarize
from src.countgdpp_inference import infer_two_pass
from src.step6_model import ContextRefiner,EncoderTap
from src.step7_inference import infer_step7,normalize_image,MODES
from src.step7_control import select_completed_checkpoint
GIB=2**30

def utc():return datetime.now(timezone.utc).isoformat()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(2**20),b''):h.update(block)
 return h.hexdigest()
def write(path,data):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def gpu_lock(args,report):
 uuid=os.environ.get('CUDA_VISIBLE_DEVICES','')
 if not uuid.startswith('GPU-') or ',' in uuid:raise ValueError('Select exactly one GPU UUID')
 import subprocess
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
 row=next(r.split(', ') for r in rows if uuid in r)
 if int(row[0])!=args.gpu_index:raise ValueError('GPU index/UUID mismatch')
 folder=Path('/tmp')/f'gpu-dispatch-{os.getuid()}';folder.mkdir(mode=0o700,exist_ok=True);handle=(folder/(uuid+'.lock')).open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
 report.update(pid=os.getpid(),hostname=socket.gethostname(),gpu_index=args.gpu_index,gpu_index_scope=args.gpu_index_scope,gpu_uuid=uuid)
 return handle

def run(args,report):
 config=json.loads(args.config.read_text());torch.set_num_threads(config['cpu_threads']);device=torch.device('cuda:0')
 if not torch.cuda.is_bf16_supported() or torch.cuda.get_device_capability(0)[0]<8 or ms_deform_attn._C is None:raise ValueError('Verified BF16/CUDA attention runtime required')
 free,total=torch.cuda.mem_get_info();required=config['allocator_limit_gib']+config['free_memory_reserve_gib']
 if free/GIB<required:raise RuntimeError('Insufficient free VRAM for step 7')
 torch.cuda.set_per_process_memory_fraction(config['allocator_limit_gib']*GIB/total)
 report.update(status='preflight',started_utc=utc(),config=config,target_images_loaded=False,gpu=torch.cuda.get_device_name(0));write(args.output/'report.json',report)
 if args.smoke:
  if not args.checkpoint:raise ValueError('Smoke needs an existing checkpoint')
  checkpoint=args.checkpoint.resolve();selection={'smoke_only':True,'checkpoint':str(checkpoint.relative_to(ROOT)),'checkpoint_sha256':sha(checkpoint)}
 else:
  if not args.series:raise ValueError('Full step 7 requires completed step-6 series')
  selection=select_completed_checkpoint(ROOT,args.series);checkpoint=ROOT/selection['checkpoint']
  if sha(checkpoint)!=selection['checkpoint_sha256']:raise ValueError('Selected checkpoint checksum mismatch')
 write(args.output/'model_selection.json',selection)
 state=torch.load(checkpoint,map_location='cpu',weights_only=True)
 if state['format']!='countgdpp_a1_v1':raise ValueError('Unsupported trained checkpoint')
 previous=state['identity']['config'];variant=previous.get('variant','A1/A2')
 for key in ('backend','bundle','short_side','max_size','num_queries','box_threshold','text_threshold','box_iou_threshold','point_radius_fraction'):
  if config[key]!=previous[key]:raise ValueError('Evaluation protocol differs from trained reference: '+key)
 for name,digest in state['identity']['code_sha256'].items():
  if name.startswith(('CountGDPlusPlus/','src/counting_data/')) or name in ('src/counting_eval.py','src/countgdpp_inference.py','src/step6_model.py'):
   if sha(ROOT/name)!=digest:raise ValueError('Selected model/evaluator source changed: '+name)
 bundle=ROOT/config['bundle'];lock=json.loads((bundle/'protocol_lock.json').read_text())
 for name,digest in lock['implementation_sha256'].items():
  if sha(ROOT/name)!=digest:raise ValueError('Frozen data code changed')
 datasets={};manifests={}
 for source in config['sources']:
  name=f'source_val_{source}.jsonl';digest=sha(bundle/name)
  if digest!=lock['artifacts_sha256'][name] or digest!=state['identity']['manifest_sha256'][name]:raise ValueError('Source validation changed')
  manifests[name]=digest;datasets[source]=UnifiedCountingDataset(bundle/name,ROOT,purpose='validate',short_side=config['short_side'],max_size=config['max_size'],text_only=True)
 cfg=SLConfig.fromfile(str(ROOT/config['model_config']));cfg.device='cuda:0';cfg.text_encoder_type=str(ROOT/'CountGD/checkpoints/bert-base-uncased');cfg.use_checkpoint=cfg.use_transformer_ckpt=False
 model,_,_=build_groundingdino(cfg);tap=None
 if variant in ('A4','A5'):
  model.occlusion_refiner=ContextRefiner(cfg.hidden_dim,previous['spatial']['hidden_channels'],variant=='A5',previous['spatial']['fixed_gate'])
  tap=EncoderTap(model,model.occlusion_refiner)
 model.load_state_dict(state['model'],strict=True);del state;gc.collect();model.to(device).eval().requires_grad_(False)
 if not Path(inspect.getfile(build_groundingdino)).resolve().is_relative_to(ROOT/'CountGDPlusPlus'):raise ValueError('Wrong backend')
 code=['scripts/run_countgdpp_step7.py','src/step7_inference.py','src/step7_control.py','src/step6_model.py','src/counting_eval.py','src/countgdpp_inference.py']
 identity={'checkpoint_sha256':selection['checkpoint_sha256'],'architecture_variant':variant,'manifest_sha256':manifests,'config_sha256':sha(args.config),'code_sha256':{name:sha(ROOT/name) for name in code},'torch':str(torch.__version__),'cuda_extension_sha256':sha(ms_deform_attn._C.__file__)};write(args.output/'identity.json',identity)
 for name in code:
  dest=args.output/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
 rows=[];seen=set();costs={mode:[] for mode in MODES};forwards=0;torch.cuda.reset_peak_memory_stats();start=time.monotonic()
 report.update(status='profiling' if args.smoke else 'evaluating',model_variant=variant,images_expected=3 if args.smoke else sum(map(len,datasets.values())));write(args.output/'report.json',report)
 output=args.output/'predictions.jsonl';stream=output.with_suffix('.jsonl.tmp').open('w')
 try:
  for source,ds in datasets.items():
   for index in ([0] if args.smoke else range(len(ds))):
    resized,target=ds[index];record=ds.record(index);path=(ROOT/record['image']).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Image path escapes workspace')
    with Image.open(path) as im:image=im.convert('RGB')
    predictions,metadata=infer_step7(model,image,target['prompt'],nested_tensor_from_tensor_list,device,config)
    if args.smoke:
     if not torch.equal(resized,normalize_image(image,config['short_side'],config['max_size'])):raise ValueError('Whole-image transform differs from baseline')
     reference=infer_two_pass(model,resized,target['prompt'],nested_tensor_from_tensor_list,device=device,pseudo_threshold=config['box_threshold'],num_exemplars=3)
     for mode in ('single','pseudo3'):
      for key in ('boxes','scores','text_min'):
       if not np.allclose(predictions[mode][key],reference[mode][key],rtol=1e-3,atol=1e-5):raise ValueError('Baseline inference changed: '+mode+'/'+key)
      if int((predictions[mode]['scores']>config['box_threshold']).sum())!=int((reference[mode]['scores']>config['box_threshold']).sum()):raise ValueError('Baseline count changed')
    image.close();modes={}
    for mode,prediction in predictions.items():
     result=evaluate_image(prediction,target,threshold=config['box_threshold'],text_threshold=config['text_threshold'],box_iou_threshold=config['box_iou_threshold'],point_radius_fraction=config['point_radius_fraction'])
     capacity=config['num_queries']*(1+metadata['crop_count']) if mode in ('crop_single','filtered_pseudo3_crop') else config['num_queries']
     result.update(query_capacity=capacity,gt_above_query_capacity=target['count']>capacity,near_query_capacity=result['raw_count']>=capacity-100,single_view_capacity_exceeded=target['count']>config['num_queries'])
     modes[mode]=result
     def used(t):
      if mode=='single':return t['view']=='full' and t['kind']=='single'
      if mode=='pseudo3':return t['view']=='full' and t['kind'] in ('single','pseudo3')
      if mode=='filtered_pseudo3':return t['view']=='full' and t['kind'] in ('single','filtered_pseudo3')
      if mode=='crop_single':return t['kind']=='single'
      return t['kind'] in ('single','filtered_pseudo3')
     costs[mode].append(sum(t['seconds'] for t in metadata['forward_timings'] if used(t)))
    if target['id'] in seen:raise ValueError('Duplicate image ID')
    seen.add(target['id']);forwards+=len(metadata['forward_timings'])
    row={'id':target['id'],'dataset':target['dataset'],'modes':modes,'inference':metadata}
    stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
    rows.append({'id':target['id'],'dataset':target['dataset'],'modes':{mode:{k:v for k,v in result.items() if k!='predictions'} for mode,result in modes.items()}})
    if args.smoke or len(rows)%25==0:
     report.update(images_completed=len(rows),forward_count=forwards,updated_utc=utc(),elapsed_seconds=time.monotonic()-start,peak_reserved_gib=torch.cuda.max_memory_reserved()/GIB)
     write(args.output/'report.json',report);print('STEP7 images',len(rows),'/',report['images_expected'],'peak GiB',round(report['peak_reserved_gib'],3),flush=True)
 finally:stream.close()
 output.with_suffix('.jsonl.tmp').replace(output)
 if len(rows)!=report['images_expected']:raise ValueError('Incomplete validation coverage')
 metrics={}
 for mode in MODES:
  per={}
  for dataset in sorted({r['dataset'] for r in rows}):
   entries=[r['modes'][mode] for r in rows if r['dataset']==dataset];item=summarize(entries)
   for group in ('density_bin','image_occlusion'):
    item['by_'+group]={k:summarize([e for e in entries if e[group]==k]) for k in sorted({e[group] for e in entries})}
   per[dataset]=item
  metrics[mode]={'per_dataset':per,'macro_dataset_mae':sum(x['mae'] for x in per.values())/len(per),'mean_forward_seconds':sum(costs[mode])/len(rows)}
 write(args.output/'metrics.json',metrics)
 if not args.smoke:
  mode=min(MODES,key=lambda m:metrics[m]['macro_dataset_mae'])
  write(args.output/'inference_selection.json',{'selected_mode':mode,'checkpoint':selection['checkpoint'],'checkpoint_sha256':selection['checkpoint_sha256'],'source_macro_mae':metrics[mode]['macro_dataset_mae'],'config_sha256':sha(args.config),'scope':'Source validation model/config selection, not unbiased target test performance','step8_or_9_started':False})
 report.update(status='profile_passed' if args.smoke else 'completed',exit_code=0,finished_utc=utc(),images_completed=len(rows),prediction_sha256=sha(output),peak_reserved_gib=torch.cuda.max_memory_reserved()/GIB)
 if tap:tap.close()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=Path('configs/step7_v1.json'));p.add_argument('--output',type=Path,required=True);p.add_argument('--series',type=Path);p.add_argument('--smoke',action='store_true');p.add_argument('--checkpoint',type=Path);p.add_argument('--gpu-index',type=int,required=True);p.add_argument('--gpu-index-scope',choices=['host','container'],default='host');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 if (args.output/'report.json').exists():p.error('Use fresh output; no automatic retry')
 report={'status':'starting','created_utc':utc()};handle=None;code=0
 try:handle=gpu_lock(args,report);run(args,report)
 except BaseException as error:
  code=1;report.update(status='oom_stopped' if isinstance(error,torch.cuda.OutOfMemoryError) else 'failed',exit_code=1,error=repr(error),traceback=traceback.format_exc(),finished_utc=utc());traceback.print_exc()
 finally:
  write(args.output/'report.json',report)
  if handle:handle.close()
 raise SystemExit(code)
if __name__=='__main__':main()
