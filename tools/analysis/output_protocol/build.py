"""Build each actual implementation with a separate binary and manifest."""
from pathlib import Path
import json,shutil,subprocess,sys
root=Path(__file__).resolve().parents[3];w=root/'local/thor-output-protocol';out=root/'local/output-protocol-binaries';out.mkdir(exist_ok=True)
variants={'baseline':(0,0,0),'early':(1,0,0),'release':(0,1,0),'async':(1,0,1),'combined':(1,1,1)}
for name,(early,release,async_) in variants.items():
 if len(sys.argv)>1 and name not in sys.argv[1:]:continue
 print('BUILD',name,flush=True)
 flags=f'-DMK_EARLY_OUTPUT_RELEASE={early} -DMK_PUBLICATION_RELEASE_ONLY={release} -DMK_ASYNC_PUBLICATION={async_}'
 with (root/f'local/output-protocol-build-{name}.txt').open('w')as log:
  subprocess.run(['bash','-c','source tools/thor/env.sh\nmake -B -C demos/low-latency-llama GPU=THOR "EXTRA_NVCCFLAGS='+flags+'"'],cwd=w,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
 dest=out/name;dest.mkdir(exist_ok=True)
 binary=next((w/'demos/low-latency-llama').glob('mk_llama*.so'))
 for f in [binary,Path(str(binary)+'.build.json')]:shutil.copy2(f,dest/f.name)
 print('DONE',name,flush=True)
