"""Source SigLIP FMHA in the full-CTA native VM."""
import ctypes as C
import json
from pathlib import Path
import statistics
import sys
import torch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'demos/pi0'))
from faithful_pi0.library import ShadowFMHA
torch.manual_seed(131);torch.set_num_threads(8)
qkv=torch.randn((3*256,3*1152),device='cuda',dtype=torch.bfloat16)
output=torch.empty((3*256,1152),device='cuda',dtype=torch.bfloat16);gold=torch.empty_like(output)
shadow=ShadowFMHA();shadow(qkv,gold);torch.cuda.synchronize()
lib=C.CDLL(str(ROOT/'local/pi0/faithful/native_fmha_vm.so'))
create=lib.faithful_vm_fmha_create;create.argtypes=[C.POINTER(C.c_void_p)]+[C.c_void_p]*2+[C.c_int]*5+[C.c_void_p]
run=lib.faithful_vm_run;run.argtypes=[C.c_void_p,C.c_void_p];lib.faithful_vm_destroy.argtypes=[C.c_void_p]
results={}
for count in (1,2,8):
    h=C.c_void_p();code=create(C.byref(h),qkv.data_ptr(),output.data_ptr(),3,256,16,72,count,torch.cuda.current_stream().cuda_stream)
    if code:raise RuntimeError(('FMHA setup',code))
    try:
        code=run(h,torch.cuda.current_stream().cuda_stream)
        if code:raise RuntimeError(('FMHA run',code))
        torch.cuda.synchronize()
        error=float((output.float()-gold.float()).square().mean().sqrt()/gold.float().square().mean().sqrt())
        assert torch.equal(output,gold),error
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):run(h,torch.cuda.current_stream().cuda_stream)
        output.fill_(float('nan'));graph.replay();torch.cuda.synchronize();assert torch.equal(output,gold)
        for _ in range(10):graph.replay()
        torch.cuda.synchronize();samples=[]
        for _ in range(10):
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();graph.replay();b.record();b.synchronize();samples.append(a.elapsed_time(b)/count)
        results[str(count)]={'source_output_exact':True,'nrmse':error,'graph_capture':True,'per_fmha_ms':statistics.median(samples),'register_policy':'static128 (bring-up)'}
        print(count,results[str(count)],flush=True)
    finally:lib.faithful_vm_destroy(h)
(ROOT/'local/pi0/faithful/fmha-vm.json').write_text(json.dumps(results,indent=2)+'\n')
