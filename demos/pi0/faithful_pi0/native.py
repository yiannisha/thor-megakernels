"""Native source-policy program: group-2 VM regions and group-1 FMHA VM nodes."""
import ctypes as C
import os
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3]

class Pointwise(C.Structure):
    _fields_=[(n,C.c_void_p) for n in ('x','a','b','c','out','k','v')]+[(n,C.c_int) for n in ('kind','m','n','kdim','ldx','ldo','p0','p1','p2','p3')]+[('scale',C.c_float)]
KINDS={name:i+10 for i,name in enumerate(('rms','layernorm','softmax','patch','rope','bias','patch_bias','bias_residual','gelu','gate','add_silu','copy','zero','embedding'))}

class NativeLibrary:
    def __init__(self,path=None):
        self.lib=C.CDLL(str(path or ROOT/'local/pi0/faithful/native_program.so'))
        specs={
            'generic_create':[C.POINTER(C.c_void_p)]+[C.c_void_p]*3+[C.c_int]*7+[C.c_float]*2+[C.c_void_p],
            'matrix_create':[C.POINTER(C.c_void_p),C.c_int]+[C.c_void_p]*3+[C.c_int]*3+[C.c_float]*2+[C.c_void_p],
            'pointwise_create':[C.POINTER(C.c_void_p),C.POINTER(Pointwise),C.c_int,C.c_void_p],
            'geglu_create':[C.POINTER(C.c_void_p)]+[C.c_void_p]*4+[C.c_int]*4+[C.c_void_p],
            'merge':[C.POINTER(C.c_void_p),C.POINTER(C.c_void_p),C.c_int,C.c_void_p],
            'run':[C.c_void_p,C.c_void_p],
            'destroy':[C.c_void_p],
        }
        for name,args in specs.items():
            fn=getattr(self.lib,'faithful_vm_'+name);fn.argtypes=args;fn.restype=None if name=='destroy' else C.c_int
    @property
    def stream(self):return torch.cuda.current_stream().cuda_stream
    def create(self,kind,*args):
        handle=C.c_void_p();code=getattr(self.lib,'faithful_vm_'+kind)(C.byref(handle),*args,self.stream)
        if code:raise RuntimeError((kind,'CUDA/setup error',code))
        return handle
    def run(self,handle):
        code=self.lib.faithful_vm_run(handle,self.stream)
        if code:raise RuntimeError(('VM launch',code))
    def generic(self,x,w,out,alpha=1.,beta=0.):
        return self.create('generic_create',x.data_ptr(),w.data_ptr(),out.data_ptr(),x.shape[0],w.shape[0],x.shape[1],x.stride(0),w.stride(0),w.stride(1),out.stride(0),alpha,beta)
    def operation(self,op):
        t=op.tensors;p=op.params
        if op.kind=='gemm':
            x,w,out=(t[n] for n in ('x','weight','out'))
            if p['tactic']=='cublas':return self.generic(x,w,out,p['alpha'],p['beta'])
            return self.create('matrix_create',p['tactic'],x.data_ptr(),w.data_ptr(),out.data_ptr(),x.shape[0],w.shape[0],x.shape[1],p['alpha'],p['beta'])
        if op.kind=='geglu':
            x,g,u,out=(t[n] for n in ('x','gate','up','out'))
            return self.create('geglu_create',x.data_ptr(),g.data_ptr(),u.data_ptr(),out.data_ptr(),x.shape[0],g.shape[0],x.shape[1],1)
        spec=Pointwise();spec.kind=KINDS[op.kind];spec.scale=p.get('eps',p.get('scale',1.))
        out=t.get('out',t.get('x',t.get('q')));spec.out=out.data_ptr();spec.m=out.numel()//out.shape[-1];spec.n=out.shape[-1];spec.ldo=out.stride(-2) if out.ndim>=2 else out.shape[-1]
        def ptr(field,name):
            if name in t:setattr(spec,field,t[name].data_ptr())
        ptr('x','x');ptr('a','weight');ptr('b','bias')
        if op.kind in ('rms','layernorm'):pass
        elif op.kind=='softmax':spec.p0=p['mask_rows'];spec.p1=p['mask_start'];spec.p2=p['pad_start']
        elif op.kind=='patch':spec.p0=t['x'].shape[0]
        elif op.kind=='rope':
            ptr('x','qkv');ptr('a','rope');ptr('out','q');ptr('k','k');ptr('v','v')
            spec.m=t['qkv'].shape[0];spec.n=t['qkv'].shape[1];spec.ldo=t['k'].shape[-1]
            spec.p0=p['qdim'];spec.p1=p['kdim'];spec.p2=p['hd'];spec.p3=p['offset']
        elif op.kind in ('bias','patch_bias','bias_residual','add_silu'):
            ptr('a','bias');ptr('b','pos' if op.kind=='patch_bias' else 'residual')
        elif op.kind=='gate':ptr('x','gate');ptr('a','up')
        elif op.kind=='copy':spec.ldx=t['x'].stride(-2)
        elif op.kind=='embedding':ptr('x','weight');ptr('a','tokens')
        return self.create('pointwise_create',C.byref(spec),1)

class NativeModel:
    def __init__(self,shadow,*,library=None,fmha_library=None):
        self.trace=os.environ.get('FAITHFUL_TRACE_REGIONS')=='1';self.shadow=shadow;self.api=NativeLibrary(library);self.parts=[];self.regions=[];self.region_ranges=[];self.graph=None
        self.fmha=C.CDLL(str(fmha_library or ROOT/'local/pi0/faithful/native_fmha_vm.so'))
        self.fmha.faithful_vm_fmha_create.argtypes=[C.POINTER(C.c_void_p)]+[C.c_void_p]*2+[C.c_int]*5+[C.c_void_p]
        self.fmha.faithful_vm_run.argtypes=[C.c_void_p,C.c_void_p];self.fmha.faithful_vm_destroy.argtypes=[C.c_void_p]
        component_starts={r[0] for r in shadow.components.values()};region=[];indices=[];limit=int(os.environ.get("FAITHFUL_REGION_LIMIT","0"))
        def flush():
            if region:
                handles=(C.c_void_p*len(region))(*(h.value for h in region))
                merged=self.api.create('merge',handles,len(region));self.regions.append(('group2',merged));self.region_ranges.append((indices[0],indices[-1]));region.clear();indices.clear()
        try:
            for i,op in enumerate(shadow.ops):
                pair_boundary=op.kind=='gemm' and op.params.get('tactic')==2
                if pair_boundary or i in component_starts:flush()
                if op.kind=='fmha':
                    flush();p=op.params;t=op.tensors;h=C.c_void_p()
                    code=self.fmha.faithful_vm_fmha_create(C.byref(h),t['qkv'].data_ptr(),t['out'].data_ptr(),p['batches'],p['rows'],p['heads'],p['hd'],1,self.api.stream)
                    if code:raise RuntimeError(('FMHA setup',code))
                    self.regions.append(('group1',h));self.region_ranges.append((i,i))
                else:
                    h=self.api.operation(op);self.parts.append(h);region.append(h);indices.append(i)
                    if pair_boundary or (limit and len(region)>=limit):flush()
                if (i+1)%500==0:print('Native source program:',i+1,'operations prepared',flush=True)
            flush()
        except Exception:self.close();raise
        self.output=shadow.output
    def run(self):
        for index,(kind,h) in enumerate(self.regions):
            if self.trace:print('REGION START',index,kind,self.region_ranges[index],flush=True)
            if kind=='group2':self.api.run(h)
            else:
                code=self.fmha.faithful_vm_run(h,self.api.stream)
                if code:raise RuntimeError(('FMHA launch',code))
            if self.trace:torch.cuda.synchronize();print('REGION PASS',index,flush=True)
        return self.output
    @torch.no_grad()
    def forward(self,images,state,noise,tokens):
        s=self.shadow;s.images.copy_(images);s.state.copy_(state.reshape_as(s.state));s.noise.copy_(noise.reshape_as(s.noise));s.tokens.copy_(tokens.reshape_as(s.tokens))
        if self.graph is None:self.run()
        else:self.graph.replay()
        return self.output
    def capture(self):
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):self.run();self.run()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.run()
    def close(self):
        self.graph=None
        for kind,h in reversed(self.regions):
            (self.api.lib if kind=='group2' else self.fmha).faithful_vm_destroy(h)
        self.regions.clear()
        for h in self.parts:self.api.lib.faithful_vm_destroy(h)
        self.parts.clear()
