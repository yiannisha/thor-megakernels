"""Bounded mixed-op VM probe; use an external process watchdog."""
import ctypes as C
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'demos/pi0'))
from faithful_pi0.native import NativeLibrary
from faithful_pi0.library import ShadowSupport
from faithful_pi0.program import Operation
api=NativeLibrary();support=ShadowSupport();torch.manual_seed(137)
x=torch.randn((768,2048),device='cuda',dtype=torch.bfloat16);weight=torch.randn((2560,2048),device='cuda',dtype=torch.bfloat16)
w=torch.ones((2048,),device='cuda',dtype=torch.bfloat16);norm=torch.empty_like(x);out=torch.empty((768,2560),device='cuda',dtype=torch.bfloat16)
rms=Operation(lambda:support.rms(x,w,norm),'rms',{'x':x,'weight':w,'out':norm},{'eps':1e-6})
qkv=Operation(lambda:None,'gemm',{'x':norm,'weight':weight,'out':out},{'tactic':0,'alpha':1.,'beta':0.})
parts=[]
for name,ops in [('rms-rms',[rms,rms]),('qkv-qkv',[qkv,qkv]),('rms-qkv',[rms,qkv]),('qkv-rms',[qkv,rms]),('rms-qkv-rms',[rms,qkv,rms])]:
    print('START',name,flush=True)
    handles=[api.operation(op) for op in ops];raw=(C.c_void_p*len(handles))(*(h.value for h in handles));h=api.create('merge',raw,len(handles))
    api.run(h);torch.cuda.synchronize();print('PASS',name,flush=True)
    api.lib.faithful_vm_destroy(h)
    for part in handles:api.lib.faithful_vm_destroy(part)
