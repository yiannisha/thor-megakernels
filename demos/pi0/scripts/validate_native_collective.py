"""Exercise native static scheduling of the source two-CTA QKV collective."""
import ctypes as C
import json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3]
torch.manual_seed(107)
a=torch.randn((768,2048),device='cuda',dtype=torch.bfloat16)
b=torch.randn((2560,2048),device='cuda',dtype=torch.bfloat16)
out=torch.empty((768,2560),device='cuda',dtype=torch.bfloat16)
lib=C.CDLL(str(ROOT/'local/pi0/faithful/native_probe.so'))
fn=lib.faithful_native_probe;fn.argtypes=[C.c_void_p]*3+[C.c_int]*3+[C.c_void_p]
code=fn(a.data_ptr(),b.data_ptr(),out.data_ptr(),768,2560,2048,torch.cuda.current_stream().cuda_stream)
if code:raise RuntimeError(('native collective',code))
gold=a@b.T
error=float((out.float()-gold.float()).square().mean().sqrt()/gold.float().square().mean().sqrt())
assert torch.isfinite(out).all() and error<.02,error
result={'native_qkv_nrmse':error,'physical_grid':[4,4,1],'cluster':[2,2,1],'workers':16,'host_gemm_launches':0,'tile_scheduler':'static resident cluster stride'}
(ROOT/'local/pi0/faithful/native-probe.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
