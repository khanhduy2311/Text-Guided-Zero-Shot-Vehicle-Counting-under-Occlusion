"""Local runtime settings; separate from the frozen semantic decision policy."""
import hashlib,json
from pathlib import Path

def load_execution():
 p=json.loads(Path(__file__).with_name('execution.json').read_text())
 if p['version']!='execution_v1' or p['residency'] not in ['auto','offload']:
  raise ValueError('Invalid execution mode')
 # Two workers are capped at 2.5 GiB each, plus at least 3 GiB free headroom.
 if not 8<=p['resident_available_gib']<=32:raise ValueError('Invalid resident GPU budget')
 if not isinstance(p['optimize_crops'],bool):raise ValueError('Invalid crop optimization setting')
 return p

def execution_sha(settings):
 return hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()

def retain_weights(free_mib,reserved_mib,settings):
 # Add back ONLY this demo's measured idle allocator reservations. This avoids
 # oscillating after our own resident models reduce the global free counter.
 return settings['residency']=='auto' and free_mib+reserved_mib>=settings['resident_available_gib']*1024
