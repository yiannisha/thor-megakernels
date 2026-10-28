"""Thor batch-one linear probe: resident weights, quantization-inclusive timing.

CUTLASS must be cloned at the revision recorded in the report before build.sh.
No production runtime imports or changes are needed.
"""
import argparse
import ctypes as C
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]

class Primitive:
    def __init__(self, library=None):
        library=Path(library or ROOT / 'local/thor-nvfp4/primitive.so').resolve()
        self.library=library
        metadata=Path(str(library)+'.json')
        if not metadata.exists(): raise RuntimeError(f'Missing build metadata: {metadata}')
        self.provenance=json.loads(metadata.read_text())
        if hashlib.sha256(library.read_bytes()).hexdigest()!=self.provenance['binary_sha256']:
            raise RuntimeError('Probe binary does not match its build metadata')
        required=['tools/analysis/nvfp4/primitive.cu']
        if library.name.startswith('persistent'):
            required+=['experimental/quantized_runtime/persistent.cu','experimental/quantized_runtime/format_storage.cuh']
        for name in required:
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=self.provenance['source_sha256'][name]:
                raise RuntimeError(f'Probe needs rebuilding after source change: {name}')
        if library.name.startswith('persistent'):
            overlay=ROOT/'local/thor-nvfp4/overlay/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp'
            if hashlib.sha256(overlay.read_bytes()).hexdigest()!=self.provenance.get('overlay_sha256'):
                raise RuntimeError('CUTLASS overlay does not match runtime build metadata')
        self.lib = C.CDLL(str(library))
        ptr, i, f = C.c_void_p, C.c_int, C.c_float
        self.lib.nv_error.restype = C.c_char_p
        self.lib.nv_sf_bytes.argtypes = [i, i]
        self.lib.nv_sf_bytes.restype = C.c_long
        self.lib.nv_quantize.argtypes = [ptr]*5+[i,i,f,ptr]
        self.lib.nv_simt.argtypes = [ptr]*5+[i,i,f,i,ptr]
        self.lib.nv_prepare.argtypes = [ptr]*5+[i,i,f,ptr]
        self.lib.nv_prepare.restype = ptr
        self.lib.nv_run.argtypes = [ptr, ptr]
        self.lib.nv_destroy.argtypes = [ptr]

    @property
    def stream(self):
        return torch.cuda.current_stream().cuda_stream

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f'CUDA/CUTLASS status {code}')

    def storage(self, source):
        rows, k = source.shape
        if source.device.type!='cuda' or source.dtype!=torch.bfloat16 or not source.is_contiguous():
            raise ValueError('Source must be contiguous device BF16')
        if rows<=0 or rows%8 or k<=0 or k%256:
            # Activations are a single row, unlike the aligned output operand.
            if rows!=1 or k<=0 or k%256: raise ValueError('Expected aligned rows and K multiple of 256')
        return dict(source=source, packed=torch.empty((rows,k//2),device='cuda',dtype=torch.uint8),
                    scales=torch.empty((rows,k//16),device='cuda',dtype=torch.uint8),
                    native=torch.zeros(self.lib.nv_sf_bytes(rows,k),device='cuda',dtype=torch.uint8),
                    decoded=torch.empty_like(source,dtype=torch.float32), tensor_scale=max(source.abs().max().item()/(6*448), 1e-12))

    def quantize(self, s, reference=False):
        self.check(self.lib.nv_quantize(*[s[n].data_ptr() for n in ('source','packed','scales','native')],
                   s['decoded'].data_ptr() if reference else 0,
                   *s['source'].shape,s['tensor_scale'],self.stream))

    def simt(self,w,x,y,packed):
        self.check(self.lib.nv_simt(w['source'].data_ptr(),w['packed'].data_ptr(),w['scales'].data_ptr(),
                x.data_ptr(), y.data_ptr(),*w['source'].shape,w['tensor_scale'],int(packed),self.stream))

    def prepare(self,w,x,y):
        handle = self.lib.nv_prepare(*[t.data_ptr() for t in (w['packed'],x['packed'],w['native'],x['native'],y)],
                         *w['source'].shape,w['tensor_scale']*x['tensor_scale'],self.stream)
        if not handle:
            raise RuntimeError(self.lib.nv_error().decode())
        return handle


def metrics(actual,expected):
    a,b=actual.float(),expected.float()
    return dict(max_abs=(a-b).abs().max().item(),relative_l2=((a-b).norm()/b.norm().clamp_min(1e-12)).item(),
                finite=bool(a.isfinite().all()))


def timings(fn, repeats, inner, eviction=None, warmup_replays=3):
    # CUDA graphs remove Python dispatch gaps for short primitive comparisons.
    for _ in range(3): fn()
    torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner): fn()
    for _ in range(warmup_replays): g.replay()
    torch.cuda.synchronize()
    samples=[]
    for _ in range(repeats):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        if eviction is not None: eviction.add_(1)
        start.record();g.replay();end.record();end.synchronize()
        samples.append(start.elapsed_time(end)/inner)
    return dict(median_ms=statistics.median(samples),samples_ms=samples,inner=inner)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--shapes',default='2048x2048,8192x2048,2048x8192,128256x2048')
    ap.add_argument('--repeats',type=int,default=30)
    ap.add_argument('--inner',type=int,default=10)
    ap.add_argument('--streaming',action='store_true',help='evict 128 MiB of L2 before each single-operation graph')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    torch.manual_seed(71)
    torch.cuda.set_device(0)
    p=Primitive()
    eviction=torch.zeros(128*1024*1024,device='cuda',dtype=torch.uint8) if args.streaming else None
    if args.streaming and args.inner!=1: raise ValueError('streaming needs --inner 1')
    report=dict(device=torch.cuda.get_device_name(),cc=torch.cuda.get_device_capability(),
                torch=torch.__version__,provenance=p.provenance,streaming=args.streaming,activation_tensor_scale='fixed, selected from input amax at setup; block scales dynamic',cutlass_revision=subprocess.check_output(
                    ['git','-C',str(ROOT/'local/cutlass-nvfp4'),'rev-parse','HEAD'],text=True).strip(),
                source_sha256=hashlib.sha256((Path(__file__).parent/'primitive.cu').read_bytes()).hexdigest(),cases=[])
    for shape in args.shapes.split(','):
        m,k=map(int,shape.split('x'))
        w=p.storage(torch.randn((m,k),device='cuda',dtype=torch.bfloat16)*0.02)
        x=p.storage(torch.randn((1,k),device='cuda',dtype=torch.bfloat16))
        p.quantize(w, reference=True);p.quantize(x, reference=True)
        y=torch.empty(m,device='cuda',dtype=torch.bfloat16)
        handle=p.prepare(w,x,y)
        # Quantized-value reference separates implementation accuracy from quantization error.
        ref_bf=(w['source'].float()@x['source'].flatten().float()).bfloat16()
        ref_w4=(w['decoded'].float()@x['source'].flatten().float()).bfloat16()
        ref_w4a4=(w['decoded'].float()@x['decoded'].flatten().float()).bfloat16()
        cases={
          'bf16_simt':lambda:p.simt(w,x['source'],y,False),
          'w4a16_simt':lambda:p.simt(w,x['source'],y,True),
          'w4a4_tcgen05_prequantized':lambda:p.check(p.lib.nv_run(handle,p.stream)),
          'w4a4_tcgen05_including_activation_quant':lambda:(p.quantize(x),p.check(p.lib.nv_run(handle,p.stream))),
          'activation_quant':lambda:p.quantize(x),
          'torch_bf16_mv':lambda:torch.mv(w['source'],x['source'].flatten(),out=y),
        }
        row=dict(m=m,k=k,weight_bytes_bf16=w['source'].numel()*2,
                 weight_bytes_nvfp4=w['packed'].numel()+w['native'].numel(),
                 quantization_error_w4=metrics(ref_w4,ref_bf),quantization_error_w4a4=metrics(ref_w4a4,ref_bf),results={})
        try:
            for name,fn in cases.items():
                fn();torch.cuda.synchronize()
                ref=ref_bf if name in ('bf16_simt','torch_bf16_mv') else ref_w4 if name=='w4a16_simt' else ref_w4a4
                error=None if name=='activation_quant' else metrics(y,ref)
                if error is not None and (not error['finite'] or error['relative_l2']>0.001):
                    raise RuntimeError(f'{shape} {name}: implementation error {error}')
                row['results'][name]=dict(**timings(fn,args.repeats,args.inner,eviction),error=error)
                print(shape,name,row['results'][name]['median_ms'],error,flush=True)
        finally:
            p.lib.nv_destroy(handle)
        report['cases'].append(row)
        args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()
