"""Check the full implementations after the timed E2E experiment finishes."""
from pathlib import Path
import json,os,subprocess
root=Path(__file__).resolve().parents[3];out=root/'docs/thor/output-protocol-results/sanitizers';out.mkdir(exist_ok=True);results=[]
for name,w,binary in [('queue_v2',root/'local/thor-output-queue-v2',root/'local/thor-output-queue-v2/demos/low-latency-llama'),('batched',root/'local/thor-output-batched',root/'local/thor-output-batched/demos/low-latency-llama')]:
 for tool in ['memcheck','synccheck','racecheck']:
  env=os.environ.copy();env['THOR_BINARY_DIR']=str(binary);env['PYTHONPATH']=str(w)
  command='source tools/thor/env.sh\ncompute-sanitizer --tool '+tool+' --target-processes all --kernel-name kns=megakernel --error-exitcode 99 python '+str(root/'tools/analysis/output_protocol/validate.py')
  previous=out/f'{name}-{tool}.txt'
  if name=='batched' and tool=='racecheck' and previous.exists() and '0 hazards displayed (0 errors, 0 warnings)' in previous.read_text():
   results.append({'variant':name,'tool':tool,'returncode':0,'reused_completed_check':True});(out/'results.json').write_text(json.dumps(results,indent=2)+'\n');continue
  print('CHECK',name,tool,flush=True)
  with (out/f'{name}-{tool}.txt').open('w')as log:
   r=subprocess.run(['bash','-c',command],cwd=w,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
  results.append({'variant':name,'tool':tool,'returncode':r.returncode});(out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
  print('DONE',name,tool,r.returncode,flush=True)
  if r.returncode:raise SystemExit(r.returncode)
