"""Check source collective reuse through real VM instruction/page lifecycles."""
import ctypes as C
import json
import os
from pathlib import Path
import statistics
import torch
ROOT=Path(__file__).resolve().parents[3]
torch.manual_seed(109)
lib=C.CDLL(os.environ.get('FAITHFUL_VM_LIBRARY',str(ROOT/'local/pi0/faithful/native_vm.so')))
create=lib.faithful_vm_create_tactic;create.argtypes=[C.POINTER(C.c_void_p),C.c_int]+[C.c_void_p]*3+[C.c_int]*4+[C.c_void_p]
run=lib.faithful_vm_run;run.argtypes=[C.c_void_p,C.c_void_p]
lib.faithful_vm_destroy.argtypes=[C.c_void_p]
results={}
for name,tactic,n in (('qkv',0,2560),('o',1,2048)):
    a=torch.randn((768,2048),device='cuda',dtype=torch.bfloat16)
    b=torch.randn((n,2048),device='cuda',dtype=torch.bfloat16)
    out=torch.empty((768,n),device='cuda',dtype=torch.bfloat16)
    for count in (1,2,8):
        handle=C.c_void_p();stream=torch.cuda.current_stream().cuda_stream
        code=create(C.byref(handle),tactic,a.data_ptr(),b.data_ptr(),out.data_ptr(),768,n,2048,count,stream)
        if code:raise RuntimeError(('VM setup',code))
        try:
            code=run(handle,stream)
            if code:raise RuntimeError(('VM run',code))
            torch.cuda.synchronize()
            gold=a@b.T
            error=float((out.float()-gold.float()).square().mean().sqrt()/gold.float().square().mean().sqrt())
            assert torch.isfinite(out).all() and error<.02,error
            for _ in range(5):run(handle,stream)
            torch.cuda.synchronize();samples=[]
            for i in range(20):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record();code=run(handle,stream);end.record();end.synchronize()
                if code:raise RuntimeError(code)
                samples.append(start.elapsed_time(end))
            # Capture/replay must preserve the resident params and instruction buffers.
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):run(handle,torch.cuda.current_stream().cuda_stream)
            out.fill_(float("nan"));graph.replay();torch.cuda.synchronize();assert torch.equal(out,gold)
            results[name+':'+str(count)]={'nrmse':error,'median_ms':statistics.median(samples),'per_gemm_ms':statistics.median(samples)/count,'graph_capture':True}
            print(name,count,results[name+':'+str(count)],flush=True)
        finally:lib.faithful_vm_destroy(handle)
(ROOT/'local/pi0/faithful/vm-validation.json').write_text(json.dumps(results,indent=2)+'\n')
