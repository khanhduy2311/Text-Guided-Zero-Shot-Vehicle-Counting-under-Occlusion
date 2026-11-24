'use strict';
const $=id=>document.getElementById(id), NS='http://www.w3.org/2000/svg';
const state={page:"count",drawer:false,file:null,example:null,imageURL:null,width:0,height:0,query:null,result:null,job:null,health:null,view:'input',running:false,selected:null,examples:[],started:0,selection:0,validation:0};
let queryTimer,toastTimer,pollTimer;const blobs=[];
const api=async(url,options={})=>{const r=await fetch(url,{...options,headers:{...options.headers}});let d;try{d=await r.json()}catch{throw Error('Không đọc được phản hồi. Hãy kiểm tra kết nối và thử lại.')}if(!r.ok){let m=typeof d.detail==='string'?d.detail:'Yêu cầu chưa hợp lệ. Hãy kiểm tra ảnh và mô tả.';const e=Error(m);e.status=r.status;throw e}return d};
function toast(message){$('toast').textContent=message;$('toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,4500)}
function error(id,message){const e=$(id);e.textContent=message||'';e.hidden=!message}
function historyRead(){try{return JSON.parse(localStorage.getItem('countlab-history')||'[]').filter(x=>x.id&&Date.now()-x.time<12*3600*1000).slice(0,8)}catch{return []}}
function historyWrite(rows){try{localStorage.setItem('countlab-history',JSON.stringify(rows.slice(0,8)))}catch{toast('Trình duyệt không cho phép lưu lịch sử. Kết quả hiện tại vẫn dùng được.')}renderHistory()}
function icon(name){const s=document.createElementNS(NS,'svg');s.classList.add('icon');s.setAttribute('aria-hidden','true');const u=document.createElementNS(NS,'use');u.setAttribute('href','#i-'+name);s.append(u);return s}
function fitImage(){
 if(!state.width||!state.height)return;const stage=$('image-stage');if(!stage.clientWidth||!stage.clientHeight)return;const style=getComputedStyle(stage),w=stage.clientWidth-parseFloat(style.paddingLeft)-parseFloat(style.paddingRight),h=stage.clientHeight-parseFloat(style.paddingTop)-parseFloat(style.paddingBottom),scale=Math.min(w/state.width,h/state.height);
 $('image-frame').style.width=state.width*scale+'px';$('image-frame').style.height=state.height*scale+'px';drawOverlay();
}
function revoke(){while(blobs.length)URL.revokeObjectURL(blobs.pop())}
function resetResult(){$('view-result').hidden=true;toggleDownload(false);state.result=null;state.job=null;state.selected=null;$('count').textContent='—';$('count-query').textContent='Chưa chạy';$('candidate-count').textContent='—';$('inference-time').textContent='—';$('object-total').textContent='0';$('object-list').replaceChildren();const p=document.createElement('p');p.className='empty-list';p.textContent='Chạy mô hình để xem từng đối tượng và điểm tương hợp.';$('object-list').append(p);$('objects-hint').textContent='Các vị trí tìm được sẽ xuất hiện tại đây.';$('score-note').hidden=true;$('download-image').disabled=true;$('download-data').disabled=true;document.querySelectorAll('[data-view]:not([data-view=input])').forEach(b=>b.disabled=true);$('display-controls').hidden=true;setView('input');syncShell()}
async function setImage(url,{filename,example=null,file=null,prompt=null}={}){
 if(state.running){toast('Hãy đợi hoặc hủy tác vụ hiện tại trước khi đổi ảnh.');return}
 const token=++state.selection;error('file-error','');const image=new Image();
 try{await new Promise((resolve,reject)=>{image.onload=resolve;image.onerror=()=>reject(Error('Không đọc được ảnh. Hãy thử một file khác.'));image.src=url})}catch(e){error('file-error',e.message);return}
 if(token!==state.selection)return;
 if(image.naturalWidth*image.naturalHeight>24_000_000||Math.min(image.naturalWidth,image.naturalHeight)<32||Math.max(image.naturalWidth,image.naturalHeight)>12000){error('file-error','Ảnh cần có cạnh từ 32 đến 12.000 px và tổng không quá 24 megapixel.');return}
 resetResult();state.imageURL=url;state.file=file;state.example=example;state.width=image.naturalWidth;state.height=image.naturalHeight;
 $('main-image').src=url;$('file-thumb').src=url;$('file-name').textContent=filename||'Ảnh đã chọn';$('file-meta').textContent=`${state.width} × ${state.height} px${file?' · '+(file.size/1024/1024).toFixed(2)+' MB':''}`;$('image-dimensions').textContent=`${state.width} × ${state.height}`;
 $('dropzone').classList.add('has-file');$('file-summary').hidden=false;$('canvas-empty').hidden=true;$('image-frame').hidden=false;$('canvas-badge').hidden=false;$('expand').disabled=false;
 if(prompt!==null)$('prompt').value=prompt;document.querySelectorAll('.example-card').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.example===example)));fitImage();validatePrompt();updateRun();
}
async function chooseFile(file){
 if(!file)return;if(!['image/jpeg','image/png','image/webp'].includes(file.type)){error('file-error','Chỉ hỗ trợ ảnh JPG, PNG và WebP.');return}if(file.size>10*1024*1024){error('file-error','Ảnh vượt quá 10 MB. Hãy chọn ảnh nhỏ hơn.');return}
 const url=URL.createObjectURL(file);blobs.push(url);await setImage(url,{file,filename:file.name});
}
function clearImage(){
 if(state.running)return;state.selection++;revoke();state.file=null;state.example=null;state.imageURL=null;state.width=state.height=0;resetResult();$('main-image').removeAttribute('src');$('file-input').value='';$('dropzone').classList.remove('has-file');$('file-summary').hidden=true;$('image-frame').hidden=true;$('canvas-empty').hidden=false;$('canvas-badge').hidden=true;$('expand').disabled=true;$('image-dimensions').textContent='';error('file-error','');document.querySelectorAll('.example-card').forEach(b=>b.setAttribute('aria-pressed','false'));updateRun();
}
async function validatePrompt(){
 clearTimeout(queryTimer);const token=++state.validation,text=$('prompt').value;$('prompt-length').textContent=text.length+'/160';state.query=null;$('query-understanding').hidden=true;updateRun();
 if(!text.trim()){error('query-error','');$('prompt').removeAttribute('aria-invalid');return}
 try{const q=await api('/api/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:text,branch:branchMenu.value})});if(token!==state.validation)return;state.query=q;$('query-canonical').textContent=q.canonical;$('query-detail').textContent=q.detail+(q.warnings?.length?' '+q.warnings.join(' '):'');$('query-understanding').hidden=false;error('query-error','');$('prompt').removeAttribute('aria-invalid');updateRun()}
 catch(e){if(token!==state.validation)return;error('query-error',e.message);$('prompt').setAttribute('aria-invalid','true');updateRun()}
}
const branchMenu=$('model-branch');
const branchNames={original:'CountGD++ gốc',a3:'CountGD++ A3 + SigLIP2',sam3:'CountGD-Box + SAM 3',a3_sam3:'CountGD++ A3 + SAM 3'};
function displayBranch(key){
 branchMenu.value=Object.hasOwn(branchNames,key)?key:'a3';
 const original=branchMenu.value==='original', hybrid=branchMenu.value==='a3_sam3', sam=branchMenu.value==='sam3', name=branchNames[branchMenu.value];
 $('side-model').textContent=name;
 $('bench-model').textContent=name;
 document.querySelector('.model-square').textContent=original?'GD++':sam?'SAM 3':'A3';
 document.querySelector('.art-label').textContent=original?'VISION ENGINE / GD++':hybrid?'VISION ENGINE / A3 + SAM 3':sam?'VISION ENGINE / SAM 3':'VISION ENGINE / A3';
 $('branch-description').textContent=original?'CountGD++ gốc nhận trực tiếp cụm mô tả và trả về số đếm.':hybrid?'A3 đề xuất xe, SAM 3 xác nhận theo mô tả.':sam?'CountGD-Box đề xuất xe, SAM 3 xác nhận theo mô tả.':'A3 định vị xe, SigLIP2 kiểm tra màu và thuộc tính.';
 if(state.allHealth)updateHealth(state.allHealth);
}
branchMenu.onchange=()=>{
 displayBranch(branchMenu.value);
 const url=new URL(location.href);
 url.searchParams.set('branch',branchMenu.value);
 history.replaceState(null,'',url);
 resetResult();error('job-error','');validatePrompt();
};
function updateRun(){const available=state.health&&['ready','busy'].includes(state.health.status);$('run').disabled=!available||!state.imageURL||!state.query||state.running;if(state.running)$('run-label').textContent='Đang xử lý';else if(state.health?.status==='starting')$('run-label').textContent='Đang nạp mô hình';else $('run-label').textContent=state.result?'Chạy lại với mô tả này':'Đếm phương tiện'}
function lockInputs(locked){state.running=locked;for(const id of ['prompt','browse','replace-file','clear-image','file-input','model-branch'])$(id).disabled=locked;document.querySelectorAll('[data-prompt],.example-card').forEach(b=>b.disabled=locked);$('canvas-loading').hidden=!locked;updateRun();syncShell()}
function updateHealth(snapshot){state.allHealth=snapshot;const h=snapshot.branches?.[branchMenu.value]||(branchMenu.value==='a3'?snapshot:{status:'offline'});state.health=h;const e=$('engine-status');e.dataset.state=h.status;const map={ready:'GPU sẵn sàng',busy:'GPU đang xử lý',starting:'Đang nạp mô hình',error:'Mô hình chưa sẵn sàng',preview:'Bản xem giao diện',offline:'Mất kết nối'};const label=map[h.status]||h.stage;if($('engine-text').textContent!==label)$('engine-text').textContent=label;updateRun()}
async function health(){try{updateHealth(await api('/api/health'))}catch{updateHealth({status:'offline'})}}
async function run(){
 if(state.running||state.page!=='count')return;await validatePrompt();if(!state.query||!state.imageURL||$('run').disabled)return;error('job-error','');toggleDownload(false);
 const form=new FormData();form.set('branch',branchMenu.value);form.set('prompt',$('prompt').value);if(state.file)form.set('image',state.file);else if(state.example)form.set('example',state.example);else {error('job-error','Hãy chọn lại ảnh để chạy một tác vụ mới.');return}
 lockInputs(true);state.started=Date.now();$('job-progress').hidden=false;$('progress-stage').textContent='Đang gửi ảnh';$('progress-bar').style.width='8%';$('queue-detail').textContent='Chuẩn bị tác vụ';$('elapsed').textContent='0s';
 try{const j=await api('/api/jobs',{method:'POST',body:form});state.job=j;pollJob(j.id)}catch(e){lockInputs(false);$('job-progress').hidden=true;error('job-error',e.message);health()}
}
async function pollJob(id){
 if(state.job?.id!==id)return;
 try{
  const j=await api('/api/jobs/'+id);if(state.job?.id!==id)return;state.job=j;$('progress-stage').textContent=j.stage;$('elapsed').textContent=Math.round((Date.now()-state.started)/1000)+'s';$('queue-detail').textContent=j.status==='queued'?`Vị trí trong hàng đợi: ${j.queue_position}`:'Đang chạy trên GPU';
  $('progress-bar').style.width=j.status==='queued'?'18%':j.stage.includes('SigLIP2')?'68%':j.stage.includes('dựng')?'90%':'38%';
  if(j.status==='completed'){lockInputs(false);$('job-progress').hidden=true;showResult(j);historyWrite([{id:j.id,branch:j.branch,model:j.result.model,prompt:j.query.canonical,count:j.result.count,time:Date.now()},...historyRead().filter(r=>r.id!==j.id)]);health();return}
  if(j.status==='error'||j.status==='cancelled'){lockInputs(false);$('job-progress').hidden=true;if(j.status==='error')error('job-error',j.error||'Không thể hoàn tất tác vụ.');else toast('Đã hủy tác vụ.');health();return}
  pollTimer=setTimeout(()=>pollJob(id),1000);
 }catch(e){if(state.job?.id!==id)return;if(e.status===404){lockInputs(false);$('job-progress').hidden=true;error('job-error',e.message);return}error('job-error','Kết nối tạm gián đoạn. Đang kiểm tra lại tác vụ đã gửi, không chạy trùng.');pollTimer=setTimeout(()=>pollJob(id),3000)}
}
function showResult(j){
 $('view-result').hidden=false;state.job=j;state.result=j.result;state.selected=null;$('count').textContent=j.result.count;$('count-query').textContent=j.query.canonical;$('candidate-count').textContent=j.result.candidate_count;$('inference-time').textContent=j.result.inference_seconds.toFixed(2)+' s';$('candidate-detail').textContent=j.result.semantic_revision?'Ứng viên trước xác minh':'Sau bước đếm cơ bản';$('object-total').textContent=j.result.count;$('objects-hint').textContent=j.result.count?'Chọn một đối tượng để làm nổi vị trí trên ảnh.':'Không tìm thấy đối tượng thỏa mô tả theo ngưỡng hiện hành.';
 $('download-image').disabled=false;$('download-data').disabled=false;$('download-json').href=`/api/jobs/${j.id}/export/json`;$('download-csv').href=`/api/jobs/${j.id}/export/csv`;$('display-controls').hidden=false;$('score-note').hidden=!j.result.semantic_revision;document.querySelectorAll('[data-view]').forEach(b=>b.disabled=false);setView('result');renderObjects();error('job-error','');updateRun();syncShell();
 if(!matchMedia('(prefers-reduced-motion: reduce)').matches)$('result-summary').animate([{opacity:.4,transform:'translateY(5px)'},{opacity:1,transform:'translateY(0)'}],{duration:260,easing:'ease-out'});
}
function setView(view){state.view=view;document.querySelectorAll('[data-view]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.view===view)));$('compare-control').hidden=view!=='compare';$('compare-line').hidden=view!=='compare';$('overlay').style.visibility=view==='input'?'hidden':'visible';$('canvas-badge').textContent=view==='input'?'Ảnh đầu vào':`${state.job?.query.canonical||''} · ${state.result?.count||0} đối tượng`;$('viewer-caption').textContent=state.result?`Kết quả cho “${state.job.query.canonical}” · ${state.job.query.pipeline}`:'Ảnh đầu vào · Chưa chạy mô hình';drawOverlay()}
function svg(tag,attrs={}){const el=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,String(v)));return el}
function drawOverlay(){
 const overlay=$('overlay');overlay.replaceChildren();overlay.setAttribute('viewBox',`0 0 ${state.width||1} ${state.height||1}`);if(!state.result)return;
 const displayWidth=$('image-frame').clientWidth||600;const scale=state.width/displayWidth,font=12*scale,stroke=2*scale;const layer=svg('g');
 if(state.view==='compare'){const defs=svg('defs'),clip=svg('clipPath',{id:'comparison-clip'});clip.append(svg('rect',{x:state.width*+$('compare-slider').value/100,y:0,width:state.width*(1-+$('compare-slider').value/100),height:state.height}));defs.append(clip);overlay.append(defs);layer.setAttribute('clip-path','url(#comparison-clip)');$('compare-line').style.left=$('compare-slider').value+'%'}
 for(const o of state.result.objects){
  const [x,y,x2,y2]=o.box;const group=svg('g',{'data-number':o.number,tabindex:0,role:'button','aria-label':`Đối tượng ${o.number}: ${state.job.query.label}`});if(state.selected&&state.selected!==o.number)group.classList.add('is-dim');if(state.selected===o.number)group.classList.add('is-highlight');
  if($('show-boxes').checked)group.append(svg('rect',{x,y,width:x2-x,height:y2-y,fill:'transparent',stroke:'#C5F66A','stroke-width':stroke,class:'main-box'}));else group.append(svg('circle',{cx:(x+x2)/2,cy:(y+y2)/2,r:4*scale,fill:'#C5F66A',stroke:'white','stroke-width':scale}));
  if($('show-labels').checked){const tx=Math.max(0,Math.min(x,state.width-35*scale)),ty=Math.max(0,y-22*scale),width=(String(o.number).length*8+12)*scale;group.append(svg('rect',{x:tx,y:ty,width,height:21*scale,rx:3*scale,fill:'#C5F66A'}));const text=svg('text',{x:tx+6*scale,y:ty+15*scale,fill:'#10170A','font-size':font,'font-family':'Be Vietnam Pro, sans-serif','font-weight':500});text.textContent=o.number;group.append(text)}
  group.addEventListener('click',()=>selectObject(o.number));group.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();selectObject(o.number)}});layer.append(group);
 }overlay.append(layer);
}
function selectObject(number){state.selected=state.selected===number?null:number;drawOverlay();document.querySelectorAll('.object-row').forEach(r=>r.setAttribute('aria-pressed',String(+r.dataset.number===state.selected)));if(state.selected)document.querySelector(`.object-row[data-number="${number}"]`)?.scrollIntoView({block:'nearest',behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'})}
function renderObjects(){
 const list=$('object-list');list.replaceChildren();const objects=state.result?.objects||[];
 if(!objects.length){const p=document.createElement('p');p.className='empty-list';p.textContent='Số đếm bằng 0. Bạn có thể thử mô tả khác hoặc một ảnh rõ hơn.';list.append(p);return}
 for(const o of objects){const row=document.createElement('div');row.className='object-row';row.dataset.number=o.number;row.setAttribute('role','button');row.tabIndex=0;row.setAttribute('aria-pressed','false');row.setAttribute('aria-label',`Làm nổi đối tượng ${o.number}`);
  const no=document.createElement('span');no.className='object-number';no.textContent='#'+o.number;
  const crop=document.createElement('img');crop.className='object-crop';crop.alt='Crop phương tiện '+o.number;crop.width=48;crop.height=38;
  try{const canvas=document.createElement('canvas');canvas.width=96;canvas.height=76;const [x,y,x2,y2]=o.box,c=canvas.getContext('2d'),scale=Math.min(96/(x2-x),76/(y2-y)),w=(x2-x)*scale,h=(y2-y)*scale;c.fillStyle='#E5E9EF';c.fillRect(0,0,96,76);c.drawImage($('main-image'),x,y,x2-x,y2-y,(96-w)/2,(76-h)/2,w,h);crop.src=canvas.toDataURL('image/jpeg',.8)}catch{crop.src=state.imageURL}
  const desc=document.createElement('div'),name=document.createElement('span'),coordinate=document.createElement('span');name.className='object-name';name.textContent=state.job.query.label;coordinate.className='object-coordinate';coordinate.textContent=`${Math.round(o.box[0])}, ${Math.round(o.box[1])} · ${Math.round(o.box[2]-o.box[0])} × ${Math.round(o.box[3]-o.box[1])} px`;desc.append(name,coordinate);
  const detector=document.createElement('div');detector.className='object-score detector-score';detector.textContent=o.detector_score.toFixed(3);const dl=document.createElement('span');dl.className='score-label';dl.textContent=state.job.branch==='original'?'Điểm CountGD++':state.job.branch==='sam3'?(o.semantic_score==null?'Điểm SAM 3':'Điểm CountGD-Box'):state.job.branch==='cubi'?'Điểm CountGD-Box':'Điểm A3';detector.append(dl);
  const sem=document.createElement('div');sem.className='object-score';sem.textContent=o.semantic_score==null?'—':o.semantic_score.toFixed(3);const sl=document.createElement('span');sl.className='score-label';sl.textContent=o.semantic_score==null?(state.job.branch==='cubi'?'Lọc mặt nạ':'Đếm cơ bản'):['sam3','a3_sam3'].includes(state.job.branch)?'Điểm SAM 3':state.job.branch==='cubi'?'SAM 2.1':'Tương hợp';sem.append(sl);row.append(no,crop,desc,detector);if(state.job.branch!=='original'&&(state.job.branch!=='sam3'||o.semantic_score!=null))row.append(sem);row.addEventListener('click',()=>selectObject(o.number));row.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();selectObject(o.number)}});list.append(row)
 }
}
function renderHistory(){const rows=historyRead(),list=$('history-list');list.replaceChildren();$('clear-history').hidden=!rows.length;if(!rows.length){const p=document.createElement('p');p.className='history-empty';p.textContent='Những kết quả bạn chạy sẽ được lưu tại đây để mở lại.';list.append(p);return}
 for(const r of rows){const wrap=document.createElement('div');wrap.className='history-entry';const b=document.createElement('button');b.type='button';b.className='history-item';const image=document.createElement('img');image.src=`/api/jobs/${r.id}/image`;image.alt='';image.loading='lazy';image.onerror=()=>image.style.visibility='hidden';const text=document.createElement('div'),name=document.createElement('strong'),meta=document.createElement('span');name.textContent=r.prompt;meta.textContent=`${r.branch==='original'?branchNames.original:r.branch==='a3_sam3'?branchNames.a3_sam3:r.branch==='sam3'?(r.model||'SAM 3'):r.branch==='cubi'?'SAM 2.1':'A3'} · ${r.count} đối tượng · ${new Date(r.time).toLocaleTimeString('vi-VN',{hour:'2-digit',minute:'2-digit'})}`;text.append(name,meta);b.append(image,text);b.onclick=()=>restore(r.id);const del=document.createElement('button');del.className='icon-button history-delete';del.type='button';del.setAttribute('aria-label','Xóa kết quả '+r.prompt);del.append(icon('trash'));del.onclick=async()=>{if(state.running&&state.job?.id===r.id)return;try{await api('/api/jobs/'+r.id,{method:'DELETE'});if(state.job?.id===r.id)resetResult()}catch(e){if(e.status!==404){toast(e.message);return}}historyWrite(historyRead().filter(x=>x.id!==r.id));toast('Đã xóa ảnh và kết quả của lần chạy này.')};wrap.append(b,del);list.append(wrap)}
}
async function restore(id){if(state.running){toast('Hãy đợi hoặc hủy tác vụ hiện tại trước khi mở lịch sử.');return}try{const j=await api('/api/jobs/'+id);if(j.status!=='completed')throw Error('Kết quả này không còn sẵn sàng.');const res=await fetch(j.input_url);if(!res.ok)throw Error('Ảnh đã hết hạn.');const blob=await res.blob(),file=new File([blob],j.filename+'.png',{type:'image/png'}),url=URL.createObjectURL(file);blobs.push(url);displayBranch(j.branch||'a3');await setImage(url,{file,filename:j.filename,prompt:j.query.original});showResult(j);navigatePage('count',{focusPanel:true});toast('Đã mở lại kết quả lưu trước đó.')}catch(e){toast(e.message);if(e.status===404)historyWrite(historyRead().filter(r=>r.id!==id))}}
function openDialog(id){$(id).showModal();$(id).querySelector('[data-close]')?.focus()}
function zoom(){if(!state.imageURL)return;const content=$('zoom-content');content.replaceChildren();const image=$('main-image').cloneNode();image.removeAttribute('id');content.append(image);if(state.result&&state.view!=='input'){const s=$('overlay').cloneNode(true);s.removeAttribute('id');s.removeAttribute('style');s.querySelectorAll('[tabindex]').forEach(g=>{g.removeAttribute('tabindex');g.removeAttribute('role')});const clip=s.querySelector('clipPath');if(clip){clip.id='zoom-comparison-clip';s.querySelector('[clip-path]')?.setAttribute('clip-path','url(#zoom-comparison-clip)')}content.append(s)}$('zoom-range').value=1;openDialog('zoom-dialog');fitZoom()}
function fitZoom(){if(!$('zoom-dialog').open)return;const v=$('zoom-viewport'),factor=Math.min((v.clientWidth-40)/state.width,(v.clientHeight-40)/state.height)*+$('zoom-range').value;$('zoom-content').style.width=state.width*factor+'px';$('zoom-content').style.height=state.height*factor+'px'}
$('browse').onclick=$('replace-file').onclick=()=>$('file-input').click();$('file-input').onchange=e=>chooseFile(e.target.files[0]);$('clear-image').onclick=clearImage;
const zone=$('dropzone');['dragenter','dragover'].forEach(n=>zone.addEventListener(n,e=>{e.preventDefault();if(!state.running)zone.classList.add('dragover')}));['dragleave','drop'].forEach(n=>zone.addEventListener(n,e=>{e.preventDefault();zone.classList.remove('dragover')}));zone.addEventListener('drop',e=>{if(state.running)return;if(e.dataTransfer.files.length!==1){error('file-error','Mỗi tác vụ dùng một ảnh. Hãy chọn một file.');return}chooseFile(e.dataTransfer.files[0])});document.addEventListener('dragover',e=>e.preventDefault());document.addEventListener('drop',e=>e.preventDefault());
$('prompt').oninput=()=>{state.query=null;++state.validation;$('prompt-length').textContent=$('prompt').value.length+'/160';$('query-understanding').hidden=true;updateRun();clearTimeout(queryTimer);queryTimer=setTimeout(validatePrompt,300)};
document.querySelectorAll('[data-prompt]').forEach(b=>b.onclick=()=>{$('prompt').value=b.dataset.prompt;validatePrompt()});$('run').onclick=run;
$('cancel-run').onclick=async()=>{if(!state.job)return;try{const j=await api('/api/jobs/'+state.job.id,{method:'DELETE'});$('progress-stage').textContent=j.stage;$('cancel-run').disabled=true;setTimeout(()=>$('cancel-run').disabled=false,2000)}catch(e){toast(e.message)}};
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>setView(b.dataset.view));$('show-labels').onchange=$('show-boxes').onchange=$('compare-slider').oninput=drawOverlay;$('expand').onclick=zoom;$('zoom-range').oninput=fitZoom;
$('download-image').onclick=()=>{if(!state.job)return;const a=document.createElement('a');a.href=`/api/jobs/${state.job.id}/image?download=true`;a.download='countlab-result.png';a.click()};function toggleDownload(open){$('download-menu').hidden=!open;$('download-data').setAttribute('aria-expanded',String(open))}
$('download-data').onclick=()=>{const open=$('download-menu').hidden;toggleDownload(open);if(open)$('download-json').focus()};document.addEventListener('click',e=>{if(!e.target.closest('.download-controls'))toggleDownload(false)});
$('view-result').onclick=()=>{$('results').focus({preventScroll:true});$('results').scrollIntoView({block:'start',behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'})};
document.querySelectorAll('[data-dialog]').forEach(b=>b.onclick=()=>openDialog(b.dataset.dialog));document.querySelectorAll('[data-close]').forEach(b=>b.onclick=()=>b.closest('dialog').close());document.querySelectorAll('dialog').forEach(d=>d.addEventListener('click',e=>{if(e.target!==d)return;const r=d.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)d.close()}));
$('clear-history').onclick=async()=>{if(state.running){toast('Hãy chờ tác vụ hiện tại hoàn tất trước khi xóa lịch sử.');return}const rows=historyRead();let failed=0;for(const r of rows){try{await api('/api/jobs/'+r.id,{method:'DELETE'})}catch(e){if(e.status!==404)failed++}}if(!failed){historyWrite([]);resetResult();toast('Đã xóa lịch sử cùng ảnh và kết quả trên server.')}else toast('Một số kết quả chưa xóa được. Hãy kiểm tra kết nối và thử lại.')};
document.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter'&&state.page==='count'&&!state.drawer&&!document.querySelector('dialog[open]')){e.preventDefault();run()}if(e.key==='Escape'&&!$('download-menu').hidden){toggleDownload(false);$('download-data').focus()}});
new ResizeObserver(fitImage).observe($('image-stage'));window.addEventListener('resize',fitZoom);

const pageNames={count:'Đếm phương tiện',examples:'Ảnh mẫu',history:'Lịch sử',method:'Phương pháp',evaluation:'Đánh giá',help:'Hướng dẫn'};
const mobileNav=matchMedia('(max-width: 960px)'),systemTheme=matchMedia('(prefers-color-scheme: dark)');
let explicitTheme=null;
try{const saved=localStorage.getItem('countlab-theme');if(['light','dark'].includes(saved))explicitTheme=saved}catch{}
function applyTheme(theme,persist=false){
 if(!['light','dark'].includes(theme))return;
 document.documentElement.dataset.theme=theme;document.documentElement.style.colorScheme=theme;
 document.querySelector('meta[name="theme-color"]').content=theme==='dark'?'#080A0B':'#F2F5EF';
 document.querySelectorAll('[data-theme-choice]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.themeChoice===theme)));
 if(persist){explicitTheme=theme;try{localStorage.setItem('countlab-theme',theme)}catch{toast('Giao diện đã đổi. Trình duyệt không cho lưu lựa chọn này.')}}
}
function syncShell(){
 $('nav-job-dot').hidden=!state.running;
 const visible=state.page!=='count'&&(state.running||state.result);
 $('current-task').hidden=!visible;
 $('current-task').textContent=state.running?'Đang đếm · Xem tác vụ':`Xem kết quả (${state.result?.count??0})`;
 $('clear-history').disabled=state.running;
}
function setDrawer(open,returnFocus=true){
 open=Boolean(open&&mobileNav.matches);state.drawer=open;
 $('sidebar').classList.toggle('is-open',open);$('sidebar').inert=mobileNav.matches&&!open;
 $('drawer-backdrop').hidden=!open;$('menu-toggle').setAttribute('aria-expanded',String(open));
 document.querySelector('.app-body').inert=open;document.body.classList.toggle('drawer-open',open);
 if(open){$('sidebar').setAttribute('role','dialog');$('sidebar').setAttribute('aria-modal','true');$('menu-close').focus()}
 else{$('sidebar').removeAttribute('role');$('sidebar').removeAttribute('aria-modal');if(returnFocus&&mobileNav.matches)$('menu-toggle').focus()}
}
function navigatePage(key,{historyMode='push',focusPanel=false,closeDrawer=true}={}){
 if(!Object.hasOwn(pageNames,key))key='count';
 state.page=key;
 document.querySelectorAll('.page-panel').forEach(p=>p.hidden=p.id!=='panel-'+key);
 document.querySelectorAll('[role="tab"][data-page]').forEach(b=>{const active=b.dataset.page===key;b.setAttribute('aria-selected',String(active));b.tabIndex=active?0:-1});
 $('page-location').textContent=pageNames[key];document.title=pageNames[key]+' · CountLab';
 if(historyMode&&location.hash!=='#'+key)window.history[historyMode==='replace'?'replaceState':'pushState'](null,'','#'+key);
 const wasDrawer=state.drawer;if(closeDrawer)setDrawer(false,false);
 toggleDownload(false);syncShell();window.scrollTo({top:0,behavior:'instant'});
 requestAnimationFrame(()=>{if(key==='count')fitImage();if(focusPanel||(wasDrawer&&closeDrawer))$('panel-'+key).focus({preventScroll:true})});
}
const pageTabs=[...document.querySelectorAll('[role="tab"][data-page]')];
pageTabs.forEach((b,i)=>{
 b.onclick=()=>navigatePage(b.dataset.page);
 b.onkeydown=e=>{let target;if(e.key==='ArrowDown')target=(i+1)%pageTabs.length;else if(e.key==='ArrowUp')target=(i+pageTabs.length-1)%pageTabs.length;else if(e.key==='Home')target=0;else if(e.key==='End')target=pageTabs.length-1;else return;e.preventDefault();pageTabs[target].focus();navigatePage(pageTabs[target].dataset.page,{closeDrawer:false})};
});
document.querySelectorAll('[data-go]').forEach(b=>b.onclick=()=>navigatePage(b.dataset.go,{focusPanel:true}));
document.querySelector('.brand').onclick=e=>{e.preventDefault();navigatePage('count',{focusPanel:true})};
document.querySelector('.skip-link').onclick=e=>{e.preventDefault();$('panel-'+state.page).focus()};
document.querySelectorAll('[data-theme-choice]').forEach(b=>b.onclick=()=>applyTheme(b.dataset.themeChoice,true));
// VOID defaults to dark. An explicit saved light/dark choice takes precedence.
window.addEventListener('storage',e=>{if(e.key==='countlab-theme'){explicitTheme=['light','dark'].includes(e.newValue)?e.newValue:null;applyTheme(explicitTheme||'dark')}});
$('menu-toggle').onclick=()=>setDrawer(!state.drawer);$('menu-close').onclick=$('drawer-backdrop').onclick=()=>setDrawer(false);
mobileNav.addEventListener('change',()=>setDrawer(false,false));
$('current-task').onclick=()=>navigatePage('count',{focusPanel:true});
window.addEventListener('popstate',()=>navigatePage(location.hash.slice(1),{historyMode:null}));
window.addEventListener('hashchange',()=>navigatePage(location.hash.slice(1),{historyMode:null}));
document.addEventListener('keydown',e=>{
 if(!state.drawer)return;
 if(e.key==='Escape'){e.preventDefault();setDrawer(false);return}
 if(e.key==='Tab'){
  const all=[...$('sidebar').querySelectorAll('a[href],button:not(:disabled)')].filter(b=>b.tabIndex>=0&&b.getClientRects().length);
  if(e.shiftKey&&document.activeElement===all[0]){e.preventDefault();all[all.length-1].focus()}
  else if(!e.shiftKey&&document.activeElement===all[all.length-1]){e.preventDefault();all[0].focus()}
 }
});
applyTheme(explicitTheme||'dark');setDrawer(false,false);
navigatePage(location.hash.slice(1)||'count',{historyMode:'replace'});
async function init(){
 displayBranch(new URLSearchParams(location.search).get('branch')==='cubi'?'sam3':new URLSearchParams(location.search).get('branch'));renderHistory();health();setInterval(health,5000);
 try{
  state.examples=await api('/api/examples');
  for(const ex of state.examples){
   const b=document.createElement('button');b.type='button';b.className='example-card';b.dataset.example=ex.id;b.setAttribute('aria-pressed','false');b.setAttribute('aria-label','Dùng ảnh mẫu '+ex.name);
   const im=document.createElement('img');im.src=ex.image;im.alt=ex.description;im.width=400;im.height=240;
   const copy=document.createElement('span');copy.className='example-copy';
   const title=document.createElement('strong');title.textContent=ex.name;
   const description=document.createElement('span');description.textContent=ex.description;
   const prompt=document.createElement('span');prompt.className='example-prompt';prompt.textContent=ex.prompt;
   const action=document.createElement('span');action.className='example-action';action.textContent='Sử dụng ảnh này';action.append(icon('arrow'));
   copy.append(title,description,prompt,action);b.append(im,copy);
   b.onclick=async()=>{if(state.running){toast('Hãy chờ tác vụ hiện tại trước khi đổi ảnh.');return}await setImage(ex.image,{example:ex.id,filename:ex.name,prompt:ex.prompt});navigatePage('count',{focusPanel:true})};
   $('examples').append(b);
  }
  const ex=state.examples[0];if(ex)await setImage(ex.image,{example:ex.id,filename:ex.name,prompt:ex.prompt});
 }catch(e){error('job-error',e.message)}
}
init();
