"""Matched warm CUDA-graph comparison: source launchers vs native VM."""
import ctypes as C
import json
from pathlib import Path
import statistics
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'demos/pi0'))
from faithful_pi0.library import ShadowGemm
lib=C.CDLL(str(ROOT/'local/pi0/faithful/native_vm.so'))
create=lib.faithful_vm_create_tactic;create.argtypes=[C.POINTER(C.c_void_p),C.c_int]+[C.c_void_p]*3+[C.c_int]*4+[C.c_void_p]
run=lib.faithful_vm_run;run.argtypes=[C.c_void_p,C.c_void_p]
lib.faithful_vm_destroy.argtypes=[C.c_void_p]
torch.manual_seed(113);torch.set_num_threads(8)
results={'protocol':'8 identical beta=0 GEMMs per graph; 0.5s alternating warmup; alternating 20-replay CUDA event batches; excludes setup and host dispatch','cases':{}}
for name,tactic,m,n,k in [('qkv',0,768,2560,2048),('o',1,768,2048,2048),('down',2,768,2048,16384)]:
    a=torch.randn((m,k),device='cuda',dtype=torch.bfloat16);b=torch.randn((n,k),device='cuda',dtype=torch.bfloat16)
    expected=a@b.T;output=torch.empty_like(expected)
    handle=C.c_void_p();code=create(C.byref(handle),tactic,a.data_ptr(),b.data_ptr(),output.data_ptr(),m,n,k,8,torch.cuda.current_stream().cuda_stream)
    if code:raise RuntimeError(('VM create',name,code))
    try:
        with ShadowGemm(a,b,tactic=tactic) as shadow:
            for _ in range(3):shadow();run(handle,torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            source_graph=torch.cuda.CUDAGraph();vm_graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(source_graph):
                for _ in range(8):shadow()
            with torch.cuda.graph(vm_graph):
                code=run(handle,torch.cuda.current_stream().cuda_stream)
                if code:raise RuntimeError(code)
            output.fill_(float('nan'));vm_graph.replay();source_graph.replay();torch.cuda.synchronize()
            assert torch.equal(output,shadow.output),name
            start=time.monotonic()
            while time.monotonic()-start<.5:
                for _ in range(20):source_graph.replay();vm_graph.replay()
                torch.cuda.synchronize()
            samples={'standalone':[],'vm':[]}
            for _ in range(10):
                for tag,graph in [('standalone',source_graph),('vm',vm_graph)]:
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(20):graph.replay()
                    end.record();end.synchronize();samples[tag].append(begin.elapsed_time(end)/160)
            med={tag:statistics.median(s) for tag,s in samples.items()}
            results['cases'][name]={'standalone_ms':med['standalone'],'vm_ms':med['vm'],'speedup':med['standalone']/med['vm'],'output_exact':True,'samples_ms':samples}
            print(name,results['cases'][name],flush=True)
    finally:lib.faithful_vm_destroy(handle)
(ROOT/'local/pi0/faithful/matched-vm.json').write_text(json.dumps(results,indent=2)+'\n')
