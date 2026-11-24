"""Container-local temporary inference state; shared NAS is for code/model artifacts."""
import hashlib,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONTROL=ROOT/'web_demo/runtime'
FAST=Path('/tmp')/('countlab-'+str(os.getuid())+'-'+hashlib.sha256(str(ROOT).encode()).hexdigest()[:10])
JOBS=FAST/'jobs'
FAST.mkdir(mode=0o700,parents=True,exist_ok=True)
JOBS.mkdir(mode=0o700,exist_ok=True)
