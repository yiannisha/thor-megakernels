"""Compare native vision arithmetic with an actual OpenPI layer fixture."""
import argparse
import ctypes as C
import json
from pathlib import Path

import torch

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('fixture',type=Path)
p.add_argument('--library',type=Path,required=True)
p.add_argument('--report',type=Path,required=True)
args=p.parse_args()
fixture={k:v.cuda() for k,v in torch.load(args.fixture,weights_only=True).items()}
library=C.CDLL(str(args.library.resolve()))
probe=library.probe_vision_math
probe.argtypes=[C.c_void_p]*4+[C.c_int]*2+[C.c_void_p]
probe.restype=C.c_int
gold=fixture['attention']
output=torch.empty_like(gold)
rows=[]
for mode in range(128):
    error=probe(*[fixture[k].data_ptr() for k in ('q','k','v')],output.data_ptr(),
                gold.shape[0],mode,torch.cuda.current_stream().cuda_stream)
    if error:raise RuntimeError(error)
    torch.cuda.synchronize()
    delta=output.float()-gold.float()
    rows.append({'mode':mode,'different_elements':int((output!=gold).sum()),
                 'nrmse':float(delta.square().mean().sqrt()/gold.float().square().mean().sqrt()),
                 'max_abs':float(delta.abs().max())})
rows.sort(key=lambda r:r['nrmse'])
args.report.write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows[:8],indent=2))
