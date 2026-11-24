"""Download pinned demo artifacts, verify SHA-256 and install atomically."""
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RELEASE = 'https://github.com/khanhduy2311/Text-Guided-Zero-Shot-Vehicle-Counting-under-Occlusion/releases/download/countlab-v1'

def valid(path, spec):
    if not path.is_file() or path.stat().st_size != spec['bytes']:
        return False
    h = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest() == spec['sha256']

def download(url, target, spec):
    if valid(target, spec):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print('Downloading', target.relative_to(ROOT), flush=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as directory:
        tmp = Path(directory)/'download'
        subprocess.run(['curl','--fail','--location','--silent','--show-error','--retry','5',
                        '--retry-all-errors','--connect-timeout','30','--output',str(tmp),url],check=True)
        if not valid(tmp,spec):
            raise RuntimeError('Checksum mismatch: '+str(target))
        os.replace(tmp,target)

def main():
    manifest = json.loads((ROOT/'web_demo/weights.json').read_text())
    folder = ROOT/'checkpoints/countlab'
    tasks = []
    for model in manifest['foundation']:
        for name,spec in model['files'].items():
            url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{name}"
            tasks.append((url,ROOT/model['target']/spec.get('local_name',name),spec))
    for name,spec in manifest['release'].items():
        tasks.append((f'{RELEASE}/{name}',folder/name,spec))
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download,*task) for task in tasks]
        for future in futures:
            future.result()
    (folder/'installed.json').write_text(json.dumps({'release':'countlab-v1'}))
    print('Verified all demo weights.')

if __name__=='__main__':
    main()
