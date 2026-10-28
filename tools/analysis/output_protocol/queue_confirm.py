"""Confirm actual E2E results against the unchanged production checkout."""
from pathlib import Path
import json,os,subprocess,time,random
root=Path(__file__).resolve().parents[3];out=root/'docs/thor/output-protocol-results/queue-v2';out.mkdir(exist_ok=True)
variants={'original':(root,root/'demos/low-latency-llama'),'queue_v2':(root/'local/thor-output-queue-v2',root/'local/thor-output-queue-v2/demos/low-latency-llama')}
order=list(variants);random.Random(311).shuffle(order);schedule=[]
for block in range(3):
 for name in order[block%len(order):]+order[:block%len(order)]:schedule.append({'block':block,'variant':name})
(out/'schedule.json').write_text(json.dumps(schedule,indent=2)+'\n')
for case in schedule:
 name=case['variant'];block=case['block'];w,binary=variants[name];env=os.environ.copy();env['THOR_BINARY_DIR']=str(binary)
 print('RUN',block,name,flush=True);started=time.time()
 with (out/f'{name}-{block}.log').open('w')as log:
  r=subprocess.run(['bash','-c','source tools/thor/env.sh\npython -m tools.thor.autoregressive_bench --warmups 5 --batches 1 --repeats 10 --decode-steps 100 --context 1024 --output '+str(out/f'{name}-{block}.json')],cwd=w,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
 case.update(returncode=r.returncode,elapsed_seconds=time.time()-started);(out/'schedule.json').write_text(json.dumps(schedule,indent=2)+'\n');print('DONE',block,name,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
