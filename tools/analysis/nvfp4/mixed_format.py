"""MX block-32 formats; setup/reference operations, never timed weight conversion."""
import torch

def native_scales(scales):
 rows,groups=scales.shape
 if rows<=0 or groups<=0 or groups%4:raise ValueError('MX scales require positive rows and a multiple of four groups')
 r=torch.arange(rows,device=scales.device)[:,None]
 g=torch.arange(groups,device=scales.device)[None,:]
 offsets=(r//128)*(groups//4)*512+(g//4)*512+(r%32)*16+(r%128//32)*4+g%4
 out=torch.zeros(((rows+127)//128)*(groups//4)*512,device=scales.device,dtype=torch.uint8)
 out[offsets]=scales
 return out

def quantize_activation(x):
 blocks=x.float().reshape(-1,32)
 exponents=torch.ceil(torch.log2((blocks.abs().amax(-1,keepdim=True)/448).clamp_min(2**-126))).clamp(-126,127).int()
 scales=torch.exp2(exponents.float())
 return (blocks/scales).to(torch.float8_e4m3fn).float().mul(scales).reshape(x.shape)

@torch.inference_mode()
def weights(source):
 if source.ndim!=2 or source.dtype!=torch.bfloat16 or not source.is_contiguous():raise ValueError('expected a contiguous BF16 matrix')
 rows,k=source.shape
 if rows<=0 or rows%8 or k<=0 or k%256:raise ValueError('expected M multiple of 8 and K multiple of 256')
 packed=torch.empty((rows,k//2),device=source.device,dtype=torch.uint8)
 scales=torch.empty((rows,k//32),device=source.device,dtype=torch.uint8)
 bounds=torch.tensor([.25,.75,1.25,1.75,2.5,3.5,5.],device=source.device)
 for start in range(0,rows,512):
  block=source[start:start+512].float().reshape(-1,k//32,32)
  e=torch.ceil(torch.log2((block.abs().amax(-1,keepdim=True)/6).clamp_min(2**-126))).clamp(-126,127)
  v=block/torch.exp2(e);a=v.abs().contiguous()
  q=torch.bucketize(a,bounds).to(torch.uint8)
  # At exact midpoints, select the even E2M1 code.
  idx=q.long().clamp_max(6)
  q=q+((q<7)&((q&1)!=0)&(a==bounds[idx])).to(torch.uint8)
  q=(q|((v<0).to(torch.uint8)*8)).reshape(-1,k)
  packed[start:start+512]=q[:,::2]|(q[:,1::2]<<4)
  scales[start:start+512]=(e.squeeze(-1)+127).to(torch.uint8)
 return dict(source=source,packed=packed,scales=scales,native=native_scales(scales),tensor_scale=1.,decoded=None)
