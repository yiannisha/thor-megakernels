"""Resident standalone plans for the BF16 FlashRT comparison target."""
import ctypes as C
import json
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[3]


class ShadowGemm:
    def __init__(self,a,b,*,tactic=0,output=None,alpha=1.,beta=0.,library=None):
        if a.dtype!=torch.bfloat16 or b.dtype!=torch.bfloat16 or a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[1]:
            raise ValueError("BF16 A[M,K] and weight[N,K] required")
        self.library=C.CDLL(str(library or ROOT/'local/pi0/faithful/shadow_gemm.so'))
        self.a=a;self.b=b;self.output=torch.empty((a.shape[0],b.shape[0]),device=a.device,dtype=a.dtype) if output is None else output
        self.handle=C.c_void_p()
        self.library.faithful_last_error.restype=C.c_char_p
        create=self.library.faithful_gemm_create
        create.argtypes=[C.POINTER(C.c_void_p),C.c_int]+[C.c_void_p]*4+[C.c_int]*3+[C.c_float]*2+[C.c_void_p]
        create.restype=C.c_int
        self.check(create(C.byref(self.handle),tactic,a.data_ptr(),b.data_ptr(),self.output.data_ptr(),self.output.data_ptr(),a.shape[0],b.shape[0],a.shape[1],alpha,beta,torch.cuda.current_stream().cuda_stream))
        self.library.faithful_gemm_run.argtypes=[C.c_void_p,C.c_void_p];self.library.faithful_gemm_run.restype=C.c_int
        self.library.faithful_gemm_destroy.argtypes=[C.c_void_p]
        self.library.faithful_gemm_traits.argtypes=[C.c_int];self.library.faithful_gemm_traits.restype=C.c_char_p
        self.manifest={**json.loads(self.library.faithful_gemm_traits(tactic)),"m":a.shape[0],"n":b.shape[0],"k":a.shape[1],"alpha":alpha,"beta":beta}

    def check(self,code):
        if code:raise RuntimeError((code,self.library.faithful_last_error().decode()))

    def __call__(self):
        self.check(self.library.faithful_gemm_run(self.handle,torch.cuda.current_stream().cuda_stream))
        return self.output

    def close(self):
        if self.handle.value:self.library.faithful_gemm_destroy(self.handle);self.handle=C.c_void_p()

    def __enter__(self):return self
    def __exit__(self,*_):self.close()


class ShadowGeGLU:
    """Two source mainloops and EVT epilogues; no setup work during replay."""
    def __init__(self,x,gate,up,output=None):
        if any(t.dtype!=torch.bfloat16 or not t.is_cuda or not t.is_contiguous() for t in (x,gate,up)):
            raise ValueError('contiguous CUDA BF16 tensors required')
        if x.ndim!=2 or gate.ndim!=2 or up.shape!=gate.shape or x.shape[1]!=gate.shape[1]:
            raise ValueError('X[M,D], gate[H,D], up[H,D] required')
        self.x=x;self.gate=gate;self.up=up
        self.output=output if output is not None else torch.empty((x.shape[0],gate.shape[0]),device=x.device,dtype=x.dtype)
        self.library=C.CDLL(str(ROOT/'local/pi0/faithful/shadow_geglu.so'))
        self.handle=C.c_void_p()
        create=self.library.faithful_geglu_create
        create.argtypes=[C.POINTER(C.c_void_p)]+[C.c_void_p]*4+[C.c_int]*3+[C.c_void_p]
        code=create(C.byref(self.handle),x.data_ptr(),gate.data_ptr(),up.data_ptr(),self.output.data_ptr(),x.shape[0],gate.shape[0],x.shape[1],torch.cuda.current_stream().cuda_stream)
        if code:raise RuntimeError(('GeGLU setup',code))
        self.library.faithful_geglu_run.argtypes=[C.c_void_p,C.c_void_p]
        self.library.faithful_geglu_destroy.argtypes=[C.c_void_p]
    def __call__(self):
        code=self.library.faithful_geglu_run(self.handle,torch.cuda.current_stream().cuda_stream)
        if code:raise RuntimeError(('GeGLU run',code))
        return self.output
    def close(self):
        if self.handle.value:self.library.faithful_geglu_destroy(self.handle);self.handle=C.c_void_p()
    def __enter__(self):return self
    def __exit__(self,*_):self.close()


class ShadowSupport:
    def __init__(self):
        self.library=C.CDLL(str(ROOT/'local/pi0/faithful/shadow_support.so'))
        specs={
            'patch_bias':[C.c_void_p]*3+[C.c_int]*2+[C.c_void_p],
            'gelu':[C.c_void_p,C.c_int,C.c_void_p],
            'gate':[C.c_void_p]*3+[C.c_int,C.c_void_p],
            'add_silu':[C.c_void_p]*2+[C.c_int,C.c_void_p],
            'bias':[C.c_void_p]*2+[C.c_int]*2+[C.c_void_p],
            'rms':[C.c_void_p]*3+[C.c_int]*2+[C.c_float,C.c_void_p],
            'layernorm':[C.c_void_p]*4+[C.c_int]*2+[C.c_float,C.c_void_p],
            'softmax':[C.c_void_p]+[C.c_int]*6+[C.c_void_p],
            'rope':[C.c_void_p]*5+[C.c_int]*7+[C.c_void_p],
            'patch':[C.c_void_p]*2+[C.c_int,C.c_void_p],
            'bias_residual':[C.c_void_p]*3+[C.c_int]*2+[C.c_void_p],
        }
        # softmax has five scalar dimensions/mask boundaries.
        specs['softmax']=[C.c_void_p]+[C.c_int]*5+[C.c_void_p]
        for name,types in specs.items():
            fn=getattr(self.library,'faithful_'+name);fn.argtypes=types;fn.restype=None
    @property
    def stream(self):return torch.cuda.current_stream().cuda_stream
    def rms(self,x,w,out,eps=1e-6):
        self.library.faithful_rms(x.data_ptr(),w.data_ptr(),out.data_ptr(),x.numel()//x.shape[-1],x.shape[-1],eps,self.stream)
    def layernorm(self,x,w,b,out,eps=1e-6):
        self.library.faithful_layernorm(x.data_ptr(),w.data_ptr(),b.data_ptr(),out.data_ptr(),x.numel()//x.shape[-1],x.shape[-1],eps,self.stream)
    def softmax(self,x,*,mask_rows=0,mask_start=None,pad_start=None):
        rows,cols=x.shape
        if cols>1024 or cols%2:raise ValueError('source attention requires even padded NK <= 1024')
        self.library.faithful_softmax(x.data_ptr(),rows,cols,mask_rows,cols if mask_start is None else mask_start,cols if pad_start is None else pad_start,self.stream)
        return x
    def patch(self,images,out):
        if tuple(images.shape[1:])!=(224,224,3):raise ValueError('NHWC 224x224x3 required')
        self.library.faithful_patch(images.data_ptr(),out.data_ptr(),images.shape[0],self.stream)
        return out
    def rope(self,qkv,rope,q,k,v,*,qdim,kdim,hd,offset=0):
        self.library.faithful_rope(qkv.data_ptr(),rope.data_ptr(),q.data_ptr(),k.data_ptr(),v.data_ptr(),qkv.shape[0],qdim,kdim,hd,qkv.shape[1],offset,k.shape[-1],self.stream)
    def bias_residual(self,x,bias,residual):
        self.library.faithful_bias_residual(x.data_ptr(),bias.data_ptr(),residual.data_ptr(),x.numel()//x.shape[-1],x.shape[-1],self.stream)
        return x

    def bias(self,x,bias):
        self.library.faithful_bias(x.data_ptr(),bias.data_ptr(),x.numel()//x.shape[-1],x.shape[-1],self.stream)
    def patch_bias(self,x,bias,pos):self.library.faithful_patch_bias(x.data_ptr(),bias.data_ptr(),pos.data_ptr(),x.shape[0],x.shape[1],self.stream)
    def gelu(self,x):self.library.faithful_gelu(x.data_ptr(),x.numel(),self.stream)
    def gate(self,g,u,out):self.library.faithful_gate(g.data_ptr(),u.data_ptr(),out.data_ptr(),out.numel(),self.stream)
    def add_silu(self,x,bias):self.library.faithful_add_silu(x.data_ptr(),bias.data_ptr(),x.numel(),self.stream)


class ShadowFMHA:
    def __init__(self):
        self.library=C.CDLL(str(ROOT/'local/pi0/faithful/shadow_fmha.so'))
        self.fn=self.library.faithful_fmha_bf16_strided
        self.fn.argtypes=[C.c_void_p]*4+[C.c_int]*8+[C.c_void_p]
    def __call__(self,qkv,out,*,batches=3,rows=256,heads=16,hd=72):
        d=heads*hd;ptr=qkv.data_ptr()
        code=self.fn(ptr,ptr+2*d,ptr+4*d,out.data_ptr(),batches,rows,rows,heads,heads,hd,3*d,3*d,torch.cuda.current_stream().cuda_stream)
        if code:raise RuntimeError(('SigLIP FMHA',code))
        return out
