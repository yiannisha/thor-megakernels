"""Validate source BF16 boundaries before full-model integration."""
import ctypes as C
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from faithful_pi0.library import ShadowSupport, ShadowGeGLU

def main():
    torch.manual_seed(103)
    support=ShadowSupport();results={}
    def rand(*shape):return torch.randn(shape,device='cuda',dtype=torch.bfloat16)
    images=rand(3,224,224,3);patches=torch.empty((768,588),device='cuda',dtype=torch.bfloat16)
    support.patch(images,patches)
    gold=images.reshape(3,16,14,16,14,3).permute(0,1,3,2,4,5).reshape(768,588)
    assert torch.equal(patches,gold);results['patch_exact']=True
    x=rand(41,1024);w=rand(1024);out=torch.empty_like(x)
    support.rms(x,w,out)
    gold=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)*w.float()).bfloat16()
    assert torch.allclose(out.float(),gold.float(),atol=.03125,rtol=.008)
    results['rms_max_error']=float((out.float()-gold.float()).abs().max())
    logits=rand(41*8,836);saved=logits.clone();support.softmax(logits,mask_rows=8,mask_start=769,pad_start=835)
    masked=saved.float();masked[:8,769:]=-float('inf');masked[8:,835:]=-float('inf')
    gold=masked.softmax(-1).bfloat16()
    assert torch.allclose(logits.float(),gold.float(),atol=1e-5,rtol=.008)
    assert torch.count_nonzero(logits[:8,769:])==0 and torch.count_nonzero(logits[:,835:])==0
    results['state_masked_softmax_max_error']=float((logits.float()-gold.float()).abs().max())
    qkv=rand(41,2560);rope=rand(41,256);q=torch.empty((41,2048),device='cuda',dtype=torch.bfloat16)
    k=torch.zeros((835,256),device='cuda',dtype=torch.bfloat16);v=torch.zeros_like(k)
    support.rope(qkv,rope,q,k,v,qdim=2048,kdim=256,hd=256,offset=768*256)
    qr=qkv[:,:2048].float().reshape(41,8,128,2);r=rope.float().reshape(41,1,128,2)
    gold=torch.stack((qr[...,0]*r[...,0]-qr[...,1]*r[...,1],qr[...,1]*r[...,0]+qr[...,0]*r[...,1]),-1).reshape_as(q).bfloat16()
    assert torch.allclose(q.float(),gold.float(),atol=.03125,rtol=.008)
    assert torch.equal(v[768:809],qkv[:,2304:]);results['rope_offset_and_v_copy']=True
    projection=rand(41,1024);bias=rand(1024);residual=rand(41,1024)
    gold=((projection.float()+bias.float()).bfloat16().float()+residual.float()).bfloat16()
    support.bias_residual(projection,bias,residual)
    assert torch.equal(projection,gold);results['vision_double_round_exact']=True
    # Real prefix dimensions exercise the source two-phase shared auxiliary EVT.
    x=rand(768,2048)*.1;gate=rand(16384,2048)*.1;up=rand(16384,2048)*.1
    with ShadowGeGLU(x,gate,up) as op:
        op();torch.cuda.synchronize()
        gf=x.float()@gate.float().T;uf=x.float()@up.float().T
        # Source gate EVT materializes post-GELU BF16 auxiliary, then up EVT
        # multiplies the FP32 up accumulator by that BF16 value.
        aux=torch.nn.functional.gelu(gf,approximate='tanh').bfloat16()
        gold=(aux.float()*uf).bfloat16()
        error=float((op.output.float()-gold.float()).square().mean().sqrt()/gold.float().square().mean().sqrt())
        assert torch.isfinite(op.output).all() and error<.015,error
        results['geglu_nrmse']=error
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):op()
        graph.replay();torch.cuda.synchronize();results['geglu_graph_capture']=True
    path=ROOT.parents[1]/'local/pi0/faithful/operators.json'
    path.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
if __name__=='__main__':main()
