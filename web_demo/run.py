"""Public HTTP surface; the only static root is web_demo/static."""
from pathlib import Path
import sys,os,json,time,io,csv,warnings,threading,re,hashlib,mimetypes
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'web_demo'
sys.path.insert(0,str(ROOT));sys.path.insert(0,os.environ.get('COUNTLAB_WEB_VENDOR',str(BASE/'vendor')))
from contextlib import asynccontextmanager
from fastapi import FastAPI,UploadFile,File,Form,HTTPException,Request
from fastapi.responses import FileResponse,JSONResponse,Response
from fastapi.staticfiles import StaticFiles
from PIL import Image,ImageOps,UnidentifiedImageError
from web_demo.query import parse_query,capabilities
from web_demo.service import Service,JOBS,RUNTIME
import uvicorn

MAX_BYTES=10*1024*1024;MAX_PIXELS=24_000_000
EXAMPLES=[{'id':'street','name':'Xe theo màu','description':'Đường phố · red car','prompt':'red car','image':'/assets/street.jpg'},
 {'id':'parking','name':'Xe bị che khuất','description':'Bãi đỗ · car','prompt':'car','image':'/assets/parking.png'},
 {'id':'riders','name':'Quan hệ người–xe','description':'Xe máy có người lái','prompt':'xe máy có người lái','image':'/assets/riders.jpg'}]
# Load public files on demand. A cold/slow NAS asset must not prevent the
# API/health endpoint from binding or force model initialization to wait.
PUBLIC_FILES={}
STATIC_ROOT=(BASE/'static').resolve()
STATIC_SIGNATURES={};STATIC_CHECKED={};STATIC_LOCK=threading.RLock()
from web_demo.local_service import LocalService
services={key:LocalService(key) for key in ('a3','original','sam3','a3_sam3')}
job_owners=services
rate={}

def select_service(branch):
 if not isinstance(branch,str) or branch not in services:raise HTTPException(422,'Nhánh mô hình không hợp lệ.')
 return services[branch]

@asynccontextmanager
async def lifespan(app):
 for owner in services.values():
  if os.environ.get('COUNTLAB_NO_GPU')!='1':owner.start()
  else:owner.status='preview';owner.stage='Chế độ kiểm tra giao diện'
 yield
 for owner in reversed(services.values()):owner.close()
app=FastAPI(title='CountLab · Vehicle Counting',docs_url=None,redoc_url=None,openapi_url=None,lifespan=lifespan)

class UploadTooLarge(Exception):pass
class RequestGuard:
 def __init__(self,app):self.app=app
 async def __call__(self,scope,receive,send):
  if scope['type']!='http':return await self.app(scope,receive,send)
  headers=dict(scope['headers']);length=headers.get(b'content-length');count=0;started=False
  async def guarded_send(message):
   nonlocal started
   if message['type']=='http.response.start':
    started=True;message['headers']=list(message.get('headers',[]))+[(b'x-content-type-options',b'nosniff'),(b'referrer-policy',b'no-referrer'),(b'x-frame-options',b'SAMEORIGIN'),(b'permissions-policy',b'camera=(), microphone=(), geolocation=()'),(b'content-security-policy',b"default-src 'self'; img-src 'self' blob: data:; font-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'self'; base-uri 'self'; form-action 'self'")]
    if scope['path'].startswith('/api/'):message['headers'].append((b'cache-control',b'no-store'))
   await send(message)
  async def reject(status,text):await JSONResponse({'detail':text},status_code=status)(scope,receive,guarded_send)
  if length:
   try:
    if int(length)>MAX_BYTES+65536:return await reject(413,'Ảnh tối đa 10 MB.')
   except ValueError:return await reject(400,'Yêu cầu không hợp lệ.')
  if scope['method'] in ['POST','DELETE'] and b'origin' in headers:
   from urllib.parse import urlsplit
   if urlsplit(headers[b'origin'].decode()).netloc!=headers.get(b'host',b'').decode():return await reject(403,'Nguồn gửi yêu cầu không hợp lệ.')
  async def bounded_receive():
   nonlocal count
   message=await receive()
   if message['type']=='http.request':
    count+=len(message.get('body',b''))
    if count>MAX_BYTES+65536:raise UploadTooLarge()
   return message
  try:await self.app(scope,bounded_receive,guarded_send)
  except UploadTooLarge:
   if not started:await reject(413,'Ảnh tối đa 10 MB.')
app.add_middleware(RequestGuard)

@app.get('/api/health')
def health():
 branches={}
 for key,owner in services.items():
  with owner.mutex:
   branches[key]={'status':owner.status,'stage':owner.stage,'queued':sum(j['status']=='queued' for j in owner.jobs.values()),'model':owner.name}
 return {**branches['a3'],'branches':branches,'max_upload_mb':10,'retention_hours':12,'capabilities':capabilities()}
@app.get('/api/examples')
def examples():return EXAMPLES
@app.post('/api/query')
async def query(request:Request):
 try:
  data=await request.json();return select_service(data.get('branch','a3')).parse_prompt(data.get('prompt',''))
 except (ValueError,AttributeError,TypeError) as e:raise HTTPException(422,str(e))
@app.post('/api/jobs',status_code=202)
async def create(request:Request,prompt:str=Form(...),image:UploadFile|None=File(default=None),example:str|None=Form(default=None),branch:str=Form(default='a3')):
 owner=select_service(branch)
 try:owner.parse_prompt(prompt)
 except ValueError as e:raise HTTPException(422,str(e))
 if owner.status=='preview':raise HTTPException(503,'Bản kiểm tra giao diện không chạy GPU. Hãy mở dịch vụ chính để suy luận.')
 client=request.client.host if request.client else 'unknown'
 if client in ['127.0.0.1','::1']:client=request.headers.get('cf-connecting-ip',client)
 now=time.monotonic()
 if now-rate.get(client,0)<2:raise HTTPException(429,'Vui lòng đợi 2 giây giữa hai lần gửi.')
 if len(rate)>1000:
  for k in list(rate):
   if now-rate[k]>60:rate.pop(k,None)
 rate[client]=now
 if bool(image)==bool(example):raise HTTPException(422,'Chọn một ảnh hoặc một ảnh mẫu.')
 if example:
  ex=next((r for r in EXAMPLES if r['id']==example),None)
  if not ex:raise HTTPException(404,'Ảnh mẫu không tồn tại.')
  raw=static_response(ex['image']).body;filename=ex['name']
 else:
  raw=await image.read(MAX_BYTES+1);filename=re.sub(r'[\x00-\x1f/\\]','_',image.filename or 'Ảnh tải lên')[:100];await image.close()
 if len(raw)>MAX_BYTES:raise HTTPException(413,'Ảnh tối đa 10 MB.')
 try:
  with warnings.catch_warnings():
   warnings.simplefilter('error',Image.DecompressionBombWarning)
   with Image.open(io.BytesIO(raw)) as src:
    if src.format not in ['PNG','JPEG','WEBP']:raise ValueError('Chỉ hỗ trợ JPG, PNG và WebP.')
    if src.width*src.height>MAX_PIXELS or min(src.size)<32 or max(src.size)>12000:raise ValueError('Ảnh cần có cạnh tối thiểu 32 px, tối đa 12.000 px và tổng không quá 24 megapixel.')
    if getattr(src,'n_frames',1)>1:raise ValueError('Vui lòng dùng ảnh tĩnh, không dùng ảnh động.')
    src.load();im=ImageOps.exif_transpose(src).convert('RGB')
 except (UnidentifiedImageError,OSError,ValueError,Image.DecompressionBombError,Image.DecompressionBombWarning) as e:raise HTTPException(422,str(e) if isinstance(e,ValueError) else 'Không đọc được ảnh. Hãy thử một file JPG/PNG/WebP hợp lệ.')
 try:return owner.create(im,prompt,filename,client)
 except OverflowError as e:raise HTTPException(429,str(e))
 except RuntimeError as e:raise HTTPException(503,str(e))
 finally:im.close()

def get_job(jid):
 if not re.fullmatch(r'[A-Za-z0-9_-]{32}',jid):raise HTTPException(404,'Không tìm thấy tác vụ.')
 j=None
 for owner in job_owners.values():
  with owner.mutex:j=owner.jobs.get(jid)
  if j:break
 if not j or time.time()-j['created_ts']>12*3600:raise HTTPException(404,'Kết quả đã hết hạn hoặc đã bị xóa. Hãy tải ảnh và chạy lại.')
 return j
@app.get('/api/jobs/{jid}')
def status(jid:str):
 j=get_job(jid);owner=job_owners[j.get('branch','a3')]
 with owner.mutex:return owner.public(j)
@app.delete('/api/jobs/{jid}')
def cancel(jid:str):
 j=get_job(jid);return job_owners[j.get('branch','a3')].cancel(jid)
@app.get('/api/jobs/{jid}/input')
def source(jid:str):
 get_job(jid);p=JOBS/jid/'input.png'
 if not p.is_file():raise HTTPException(404,'Ảnh đã bị xóa.')
 return FileResponse(p,media_type='image/png')
@app.get('/api/jobs/{jid}/image')
def output(jid:str,download:bool=False):
 j=get_job(jid)
 if j['status']!='completed':raise HTTPException(409,'Kết quả chưa sẵn sàng.')
 return FileResponse(JOBS/jid/'result.png',media_type='image/png',filename='countlab-result.png' if download else None)
@app.get('/api/jobs/{jid}/export/{fmt}')
def export(jid:str,fmt:str):
 j=get_job(jid)
 if j['status']!='completed':raise HTTPException(409,'Kết quả chưa sẵn sàng.')
 if fmt=='json':return FileResponse(JOBS/jid/'result.json',media_type='application/json',filename='countlab-result.json')
 if fmt!='csv':raise HTTPException(404,'Định dạng không được hỗ trợ.')
 s=io.StringIO();w=csv.writer(s);w.writerow(['number','query','x1','y1','x2','y2','detector_score','semantic_score'])
 query_text=j['query']['canonical']
 if query_text.lstrip().startswith(('=','+','-','@')):query_text="'"+query_text
 for o in j['result']['objects']:w.writerow([o['number'],query_text,*[round(x,3) for x in o['box']],o['detector_score'],o.get('semantic_score','')])
 return Response('\ufeff'+s.getvalue(),media_type='text/csv; charset=utf-8',headers={'Content-Disposition':'attachment; filename="countlab-objects.csv"'})

def static_response(path):
 if not (path in ['/', '/app.js', '/style.css', '/favicon.svg'] or path.startswith('/assets/')):
  raise HTTPException(404,'Không tìm thấy tài nguyên.')
 now=time.monotonic()
 with STATIC_LOCK:
  if path not in PUBLIC_FILES or now-STATIC_CHECKED.get(path,0)>2:
   file=(STATIC_ROOT/('index.html' if path=='/' else path.lstrip('/'))).resolve()
   if not file.is_relative_to(STATIC_ROOT):raise HTTPException(404,'Không tìm thấy tài nguyên.')
   try:
    if not file.is_file():raise HTTPException(404,'Không tìm thấy tài nguyên.')
    stat=file.stat();signature=(stat.st_mtime_ns,stat.st_size)
    if signature!=STATIC_SIGNATURES.get(path):
     body=file.read_bytes();after=file.stat()
     if signature==(after.st_mtime_ns,after.st_size) and len(body)==after.st_size:
      PUBLIC_FILES[path]=body;STATIC_SIGNATURES[path]=signature
   except OSError:
    pass
   STATIC_CHECKED[path]=now
  if path not in PUBLIC_FILES:raise HTTPException(404,'Không tìm thấy tài nguyên.')
  body=PUBLIC_FILES[path]
 mime=mimetypes.guess_type(path if path!='/' else 'index.html')[0] or 'application/octet-stream'
 return Response(body,media_type=mime,headers={'Cache-Control':'no-cache'})
@app.get('/assets/{name:path}')
def asset(name:str):return static_response('/assets/'+name)
@app.get('/')
def index():return static_response('/')
@app.get('/app.js')
def js():return static_response('/app.js')
@app.get('/style.css')
def css():return static_response('/style.css')
@app.get('/favicon.svg')
def icon():return static_response('/favicon.svg')
if __name__=='__main__':
 port=int(os.environ.get('COUNTLAB_PORT','7860'));RUNTIME.mkdir(exist_ok=True);(RUNTIME/'api.pid').write_text(str(os.getpid()))
 uvicorn.run(app,host=os.getenv('COUNTLAB_BIND','127.0.0.1'),port=port,workers=1,log_level='warning',access_log=False,timeout_keep_alive=15,limit_concurrency=48,timeout_graceful_shutdown=20)
