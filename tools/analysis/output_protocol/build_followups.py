"""Rebuild the tested batching and mbarrier queue implementations."""
from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parents[3]
variants={'batched':('thor-output-batched','-DMK_BATCHED_PUBLICATION=1 -DMK_EARLY_OUTPUT_RELEASE=0 -DMK_PUBLICATION_RELEASE_ONLY=0 -DMK_ASYNC_PUBLICATION=0'),'queue-v2':('thor-output-queue-v2','-DMK_EARLY_OUTPUT_RELEASE=1 -DMK_PUBLICATION_RELEASE_ONLY=0 -DMK_ASYNC_PUBLICATION=1')}
for name,(folder,flags)in variants.items():
 if len(sys.argv)>1 and name not in sys.argv[1:]:continue
 with (root/f'local/output-protocol-build-{name}.txt').open('w')as log:
  subprocess.run(['bash','-c','source tools/thor/env.sh\nmake -B -C demos/low-latency-llama GPU=THOR "EXTRA_NVCCFLAGS='+flags+'"'],cwd=root/'local'/folder,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
 print('DONE',name,flush=True)
