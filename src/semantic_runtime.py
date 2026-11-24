"""Small shared runtime helpers; no model imports, no annotation dependencies."""
import json,hashlib,os,fcntl,socket,subprocess
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
def utc():return datetime.now(timezone.utc).isoformat()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def write(path,value):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def local_path(value):
 path=Path(value)
 if path.is_absolute() or '..' in path.parts:raise ValueError('Use a project-relative path')
 resolved=(ROOT/path).resolve()
 if not resolved.is_relative_to(ROOT):raise ValueError('Path escapes the project')
 return resolved
def inherited_lock():
 uuid=os.environ.get('CUDA_VISIBLE_DEVICES','');fd=os.environ.get('GPU_DISPATCH_LOCK_FD')
 if not uuid.startswith('GPU-') or ',' in uuid or fd is None:raise ValueError('Start with run_semantic_counting.py to reserve one GPU')
 path=Path('/tmp')/f'gpu-dispatch-{os.getuid()}'/(uuid+'.lock');handle=os.fdopen(os.dup(int(fd)),'a')
 if os.fstat(handle.fileno()).st_ino!=path.stat().st_ino:raise ValueError('GPU lock FD mismatch')
 fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);return handle
