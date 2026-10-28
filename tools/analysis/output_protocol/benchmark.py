"""Serialized randomized blocks of full uninstrumented autoregressive requests."""
from pathlib import Path
import json,os,random,subprocess,time
root=Path(__file__).resolve().parents[3];w=root/'local/thor-output-protocol';out=root/'docs/thor/output-protocol-results';out.mkdir(exist_ok=True)
variants=['baseline','early','release','async','combined'];rng=random.Random(2718);schedule=[]
for block in range(3):
 order=variants.copy();rng.shuffle(order)
 for name in order:schedule.append({'block':block,'variant':name})
(out/'schedule.json').write_text(json.dumps(schedule,indent=2)+'\n')
for case in schedule:
 name=case['variant'];block=case['block'];dest=out/f'{name}-{block}.json'
 env=os.environ.copy();env['THOR_BINARY_DIR']=str(root/'local/output-protocol-binaries'/name)
 print('RUN',block,name,flush=True);started=time.time()
 with (out/f'{name}-{block}.log').open('w')as log:
  r=subprocess.run(['bash','-c','source tools/thor/env.sh\npython -m tools.thor.autoregressive_bench --warmups 5 --batches 1 --repeats 10 --decode-steps 100 --context 1024 --output '+str(dest)],cwd=w,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
 case.update(returncode=r.returncode,elapsed_seconds=time.time()-started)
 (out/'schedule.json').write_text(json.dumps(schedule,indent=2)+'\n')
 print('DONE',block,name,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
