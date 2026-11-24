"""Bounded inference queue and private artifact lifecycle for the public demo."""
from pathlib import Path
import os,sys,json,time,secrets,threading,queue,subprocess,fcntl,signal,shutil,hashlib,traceback
from datetime import datetime,timezone
from PIL import Image,ImageDraw,ImageFont
from web_demo.query import parse_query,spatial_match,needs_category_verification
from web_demo.runtime_paths import CONTROL,FAST,JOBS
from web_demo.open_text import PromptLengthError

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'web_demo';RUNTIME=CONTROL
TTL=12*3600
def utc():return datetime.now(timezone.utc).isoformat()
def write(path,data):
 path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(path)
def relative(path):
 path=Path(path)
 try:return str(path.relative_to(ROOT))
 except ValueError:
  path.relative_to(FAST)  # Private runtime only, never an HTTP-supplied path.
  return str(path)
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

class Service:
 branch='a3'
 name='CountGD++ A3 + SigLIP2'
 infer_stage='A3 đang tìm phương tiện'
 parse_prompt=staticmethod(parse_query)
 def __init__(self):
  self.jobs={};self.mutex=threading.RLock();self.pending=queue.Queue(maxsize=8);self.status='starting';self.stage='Đang kiểm tra mô hình';self.workers=[];self.lock=None;self.stop_event=threading.Event();self.current=None;self.started=time.time()
  for f in JOBS.glob('*/state.json'):
   try:
    j=json.loads(f.read_text())
    if j.get('branch','a3')!=self.branch:continue
    if time.time()-j['created_ts']>TTL:shutil.rmtree(f.parent);continue
    if j['status'] not in ['completed','error','cancelled']:j.update(status='error',error='Phiên dịch vụ trước đã kết thúc. Hãy chạy lại khi mô hình sẵn sàng.');write(f,j)
    self.jobs[j['id']]=j
   except (ValueError,OSError,KeyError):continue
 def save(self,j):write(JOBS/j['id']/'state.json',j)
 def start(self):threading.Thread(target=self.loop,name='countlab-inference',daemon=True).start()
 def cleanup(self):
  with self.mutex:
   expired=[k for k,j in self.jobs.items() if time.time()-j['created_ts']>TTL and j['status'] in ['completed','error','cancelled']]
   for k in expired:self.jobs.pop(k,None);shutil.rmtree(JOBS/k,ignore_errors=True)
 def public(self,j):
  out={k:v for k,v in j.items() if k not in ['client','created_ts','image_path','cancel_requested']}
  out['input_url']=f'/api/jobs/{j["id"]}/input';out['queue_position']=0
  if j['status']=='queued':
   waiting=[x for x in self.jobs.values() if x['status']=='queued'];out['queue_position']=1+sum(x['created_ts']<j['created_ts'] for x in waiting)
  return out
 def create(self,image,prompt,filename,client):
  q=self.parse_prompt(prompt);self.cleanup()
  with self.mutex:
   if self.status in ['error','stopped']:raise RuntimeError('Mô hình hiện chưa sẵn sàng. Vui lòng thử lại sau khi dịch vụ được khôi phục.')
   if self.pending.full():raise OverflowError('Hàng đợi đang đầy. Vui lòng đợi một lát rồi gửi lại.')
   if sum(j.get('client')==client and j['status'] in ['queued','running'] for j in self.jobs.values())>=2:raise OverflowError('Bạn đang có hai tác vụ. Hãy chờ một tác vụ hoàn tất.')
   if sum(j['status']!='cancelled' for j in self.jobs.values())>=120:raise OverflowError('Kho kết quả tạm đang đầy. Vui lòng xóa các kết quả cũ hoặc thử lại sau.')
   jid=secrets.token_urlsafe(24);folder=JOBS/jid;folder.mkdir(mode=0o700);path=folder/'input.png';image.save(path)
   j={'id':jid,'branch':self.branch,'status':'queued','stage':'Đang chờ đến lượt','created_at':utc(),'created_ts':time.time(),'client':client,'filename':filename,'query':q,'width':image.width,'height':image.height,'image_path':relative(path),'image_sha256':digest(path),'cancel_requested':False}
   self.jobs[jid]=j;self.save(j);self.pending.put_nowait(jid);return self.public(j)
 def cancel(self,jid):
  with self.mutex:
   j=self.jobs.get(jid)
   if not j:raise KeyError(jid)
   j['cancel_requested']=True
   if j['status']=='running':j['stage']='Đang hủy tác vụ'
   else:
    j.update(status='cancelled',stage='Đã xóa kết quả');j.pop('result',None)
    for p in (JOBS/jid).iterdir():
     if p.name!='state.json':p.unlink(missing_ok=True)
   self.save(j);return self.public(j)
 def cancelled(self,j):
  if not j['cancel_requested']:return False
  with self.mutex:
   j.update(status='cancelled',stage='Đã hủy tác vụ');self.save(j)
   for p in (JOBS/j['id']).iterdir():
    if p.name!='state.json':p.unlink(missing_ok=True)
  return True
 def annotate(self,j,objects):
  with Image.open(ROOT/j['image_path']) as raw:im=raw.convert('RGB')
  draw=ImageDraw.Draw(im);scale=max(1,min(im.size)/600);font=ImageFont.truetype(__import__('io').BytesIO(self.font_bytes),max(14,round(16*scale)));line=max(2,round(2.4*scale))
  for i,o in enumerate(objects,1):
   b=o['box'];draw.rectangle(b,outline='#F36B45',width=line);label=str(i);x=max(0,min(b[0],im.width-40));y=max(0,b[1]-24*scale);bb=draw.textbbox((x+5,y+2),label,font=font);draw.rounded_rectangle((x,y,max(x+26*scale,bb[2]+6),bb[3]+5),radius=4*scale,fill='#D94625');draw.text((x+5,y+2),label,font=font,fill='white')
  im.save(JOBS/j['id']/'result.png');im.close()
 def loop(self):
  try:self.bootstrap()
  except BaseException:
   traceback.print_exc()
   with self.mutex:
    self.status='error';self.stage='Mô hình chưa sẵn sàng'
    for job in self.jobs.values():
     if job['status']=='queued':job.update(status='error',stage=self.stage,error=self.stage);self.save(job)
   self.close_workers();return
  last_cleanup=time.monotonic()
  while not self.stop_event.is_set():
   if time.monotonic()-last_cleanup>60:self.cleanup();last_cleanup=time.monotonic()
   try:jid=self.pending.get(timeout=1)
   except queue.Empty:continue
   j=self.jobs[jid]
   if self.cancelled(j):self.pending.task_done();continue
   self.current=jid;self.status='busy';start=time.monotonic()
   try:
    with self.mutex:j.update(status='running',stage=self.infer_stage,started_at=utc());self.save(j)
    outputs=self.infer(j)
    if outputs is None or self.cancelled(j):continue
    d,sem,execution=outputs
    objects=[o for o in (sem or d)['objects'] if spatial_match(o['box'],(j['width'],j['height']),j['query'].get('spatial',[]))];objects=sorted(objects,key=lambda o:(o['box'][0],o['box'][1]));objects=[{**o,'number':i+1,'label':j['query']['label']} for i,o in enumerate(objects)]
    with self.mutex:j['stage']='Đang dựng kết quả';self.save(j)
    self.annotate(j,objects)
    if self.cancelled(j):continue
    result={'branch':self.branch,'count':len(objects),'candidate_count':sem['candidates'] if sem else len(objects),'objects':objects,'inference_seconds':round(d['seconds']+(sem or {}).get('seconds',0),3),'processing_seconds':round(time.monotonic()-start,3),'detector_seconds':round(d['seconds'],3),'semantic_seconds':round((sem or {}).get('seconds',0),3),'model':j['query']['pipeline'],'image_url':f'/api/jobs/{jid}/image','checkpoint_sha256':d['checkpoint_sha256'],'semantic_revision':(sem or {}).get('semantic_revision'),'annotations_used':False,'recipe':sem.get('policy',{}).get('version','full800 + frozen semantic policy') if sem else 'filtered_pseudo3','prompt_trace':{'original':j['query']['original'],'semantic':j['query'].get('semantic_prompt'),'detector':d.get('detector_prompts'),'spatial':j['query'].get('spatial',[]),**(sem or {}).get('prompt_trace',{})},'policy_sha256':(sem or {}).get('policy_sha256'),'note':'Số đếm là dự đoán. Điểm mô hình không phải độ chính xác được bảo đảm.'}
    result['execution']={'mode':'local GPU','optimize_crops':execution['optimize_crops'],'settings_sha256':execution['settings_sha256'],'detector_residency':d['gpu_residency'],'semantic_residency':(sem or {}).get('gpu_residency'),'detector_host':d.get('execution_host','localhost'),'detector_precision':d.get('precision','bfloat16'),'semantic_host':(sem or {}).get('execution_host','localhost')}
    if sem is None:result['execution']['semantic_host']=None
    if 'physical_gpu_index' in d:result['execution']['physical_gpu_index']=d['physical_gpu_index']
    if sem and 'physical_gpu_index' in sem:result['execution']['semantic_gpu_index']=sem['physical_gpu_index']
    write(JOBS/jid/'result.json',{'query':j['query'],'image':{'width':j['width'],'height':j['height'],'sha256':j['image_sha256']},'result':result,'completed_at':utc()})
    write(JOBS/jid/'performance.json',{'execution':execution,'detector':{k:v for k,v in d.items() if k!='objects'},'semantic':{k:v for k,v in (sem or {}).items() if k!='objects'}})
    with self.mutex:
     if self.cancelled(j):continue
     j.update(status='completed',stage='Đã hoàn tất',completed_at=utc(),result=result);self.save(j)
   except BaseException as e:
    traceback.print_exc();fatal='memory' in str(e).lower() or 'worker' in str(e).lower() or isinstance(e,(queue.Empty,BrokenPipeError))
    with self.mutex:j.update(status='error',stage='Không thể hoàn tất',error=str(e) if isinstance(e,PromptLengthError) else 'Suy luận gặp lỗi. Hệ thống đã dừng tác vụ; ảnh và mô tả của bạn vẫn được giữ để kiểm tra.');self.save(j)
    if fatal:
     self.status='error';self.stage='Mô hình cần được khôi phục';self.close_workers()
     with self.mutex:
      for other in self.jobs.values():
       if other['status']=='queued':other.update(status='error',error='Dịch vụ đã dừng sau lỗi GPU; không tự chạy lại.',stage='Tác vụ đã dừng');self.save(other)
     return
   finally:
    self.current=None;self.pending.task_done()
    if self.status!='error':self.status='ready';self.stage='Sẵn sàng'
 def close_workers(self):
  for w in self.workers:w.close()
  self.workers=[]
  if self.lock:self.lock.close();self.lock=None
 def close(self):self.stop_event.set();self.status='stopped';self.close_workers()
