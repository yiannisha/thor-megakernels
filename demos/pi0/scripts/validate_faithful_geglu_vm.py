"""Compare native source GeGLU against its resident standalone plan."""
import ctypes as C
import json
import os
from pathlib import Path
import statistics
import sys
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'demos/pi0'))
from faithful_pi0.library import ShadowGeGLU
torch.manual_seed(127);torch.set_num_threads(8)
lib=C.CDLL(os.environ.get('FAITHFUL_VM_LIBRARY',str(ROOT/'local/pi0/faithful/native_vm.so')))
create=lib.faithful_vm_geglu_create;create.argtypes=[C.POINTER(C.c_void_p)]+[C.c_void_p]*4+[C.c_int]*4+[C.c_void_p]
run=lib.faithful_vm_run;run.argtypes=[C.c_void_p,C.c_void_p]
lib.faithful_vm_destroy.argtypes=[C.c_void_p]
x=torch.randn((768,2048),device='cuda',dtype=torch.bfloat16)*.1
gate=torch.randn((16384,2048),device='cuda',dtype=torch.bfloat16)*.1
up=torch.randn_like(gate)*.1
out=torch.empty((768,16384),device='cuda',dtype=torch.bfloat16)
results={}
with ShadowGeGLU(x,gate,up) as shadow:
    shadow();torch.cuda.synchronize()
    for count in (1,2,8):
        h=C.c_void_p();code=create(C.byref(h),x.data_ptr(),gate.data_ptr(),up.data_ptr(),out.data_ptr(),768,16384,2048,count,torch.cuda.current_stream().cuda_stream)
        if code:raise RuntimeError(('GeGLU setup',code))
        try:
            code=run(h,torch.cuda.current_stream().cuda_stream)
            if code:raise RuntimeError(('GeGLU run',code))
            torch.cuda.synchronize()
            error=float((out.float()-shadow.output.float()).square().mean().sqrt()/shadow.output.float().square().mean().sqrt())
            assert torch.equal(out,shadow.output),error
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):run(h,torch.cuda.current_stream().cuda_stream)
            out.fill_(float('nan'));graph.replay();torch.cuda.synchronize();assert torch.equal(out,shadow.output)
            for _ in range(10):graph.replay()
            torch.cuda.synchronize();samples=[]
            for _ in range(10):
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();graph.replay();b.record();b.synchronize();samples.append(a.elapsed_time(b)/count)
            results[str(count)]={'source_output_exact':True,'nrmse':error,'graph_capture':True,'per_geglu_ms':statistics.median(samples)}
            print(count,results[str(count)],flush=True)
        finally:lib.faithful_vm_destroy(h)
(ROOT/'local/pi0/faithful/geglu-vm.json').write_text(json.dumps(results,indent=2)+'\n')
