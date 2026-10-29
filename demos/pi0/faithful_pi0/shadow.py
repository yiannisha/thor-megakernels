"""BF16 FlashRT policy oracle, using source kernels and cuBLAS projections.

This is the standalone comparison target, not the native VM implementation.
All transformer scratch is reused; source CUTLASS plans are resident at setup.
"""
from contextlib import ExitStack
import math
import torch
from .program import Operation
from .library import ShadowSupport, ShadowGemm, ShadowGeGLU, ShadowFMHA


def interleave(weight,heads):
    n,k=weight.shape
    return weight.reshape(heads,2,n//heads//2,k).permute(0,2,1,3).reshape(n,k)


class BF16Shadow:
    def __init__(self,model,*,num_views=3,prompt_len=0,chunk_size=50,flow_steps=10):
        self.model=model;self.device=next(model.parameters()).device
        self.views=num_views;self.prompt=prompt_len;self.chunk=chunk_size;self.steps=flow_steps
        self.support=ShadowSupport();self.fmha=ShadowFMHA();self.owners=ExitStack()
        self.ops=[];self.buffers={};self.weights={};self.tactics=[];self.components={}
        self.images=self.buffer('images',(num_views,224,224,3))
        self.state=self.buffer('state',(1,32));self.noise=self.buffer('noise',(chunk_size,32))
        self.tokens=self.buffer('tokens',(prompt_len,),torch.int64)
        self.graph=None
        with torch.no_grad():self.build()

    def buffer(self,name,shape,dtype=torch.bfloat16):
        if name not in self.buffers:self.buffers[name]=torch.zeros(shape,device=self.device,dtype=dtype)
        t=self.buffers[name]
        if tuple(t.shape)!=tuple(shape) or t.dtype!=dtype:raise ValueError(('scratch contract',name,t.shape,shape))
        return t

    def weight(self,name,x):
        if name not in self.weights:self.weights[name]=x.detach().to(torch.bfloat16).contiguous()
        return self.weights[name]

    def op(self,fn,kind='unclassified',tensors=None,**params):
        self.ops.append(Operation(fn,kind,tensors or {},params))
    def copy(self,out,x):self.op(lambda:out.copy_(x),'copy',{'x':x,'out':out})
    def zero(self,out):self.op(lambda:out.zero_(),'zero',{'out':out})
    def bias(self,x,b):
        self.op(lambda:self.support.bias(x,b),'bias',{'x':x,'bias':b})
    def mm(self,x,w,out,*,beta=0.,alpha=1.):
        # cuBLAS source sites: one accumulator alpha/beta and one BF16 cast.
        self.op(lambda:torch.addmm(out,x,w.T,beta=beta,alpha=alpha,out=out),'gemm',{'x':x,'weight':w,'out':out},tactic='cublas',alpha=alpha,beta=beta)
    def cutlass(self,x,w,out,tactic,beta=0.):
        plan=self.owners.enter_context(ShadowGemm(x,w,output=out,tactic=tactic,beta=beta))
        self.tactics.append(plan.manifest);self.op(plan,'gemm',{'x':x,'weight':w,'out':out},tactic=tactic,alpha=1.,beta=beta)
    def norm(self,x,module,out,*,rms=False,folded=False):
        w=self.weight(('norm',id(module),folded),torch.ones_like(module.weight) if folded else (1.+module.weight.float()) if rms else module.weight)
        if rms:self.op(lambda:self.support.rms(x,w,out),'rms',{'x':x,'weight':w,'out':out},eps=1e-6)
        else:
            b=self.weight(('normbias',id(module)),module.bias)
            self.op(lambda:self.support.layernorm(x,w,b,out,eps=module.eps),'layernorm',{'x':x,'weight':w,'bias':b,'out':out},eps=module.eps)
    def rope(self,rows,start,hd,theta):
        angle=torch.arange(start,start+rows,device=self.device,dtype=torch.float32)[:,None]*torch.pow(float(theta),-torch.arange(0,hd,2,device=self.device,dtype=torch.float32)/hd)
        return torch.stack((angle.cos(),angle.sin()),-1).reshape(rows,hd).bfloat16()

    def gemma(self,x,layer,index,rope,caches,*,prefix=False,last=False):
        rows,dim=x.shape;tag='prefix' if prefix else 'expert';att=layer.self_attn
        norm=self.buffer(tag+'.norm',(rows,dim));qkv=self.buffer(tag+'.qkv',(rows,2560))
        q=self.buffer(tag+'.q',(rows,2048));attout=self.buffer(tag+'.attout',(rows,2048))
        k,v=caches;nk=(rows if prefix else k.shape[0]-1)  # caches reserve one even-padding row
        nk_pad=(nk+1)//2*2;logits=self.buffer(tag+'.logits',(rows*8,nk_pad))
        self.norm(x,layer.input_layernorm,norm,rms=True,folded=True)
        key=(tag,index,'qkv')
        if key not in self.weights:
            scale=1.+layer.input_layernorm.weight.float()
            packed=torch.cat((interleave(att.q_proj.weight.float(),8),interleave(att.k_proj.weight.float(),1),att.v_proj.weight.float()))*scale
            self.weight(key,packed)
        qw=self.weights[key]
        if prefix:self.cutlass(norm,qw,qkv,0)
        else:self.mm(norm,qw,qkv)
        offset=0 if prefix else k.shape[0]-1-rows
        self.op(lambda:self.support.rope(qkv,rope,q,k,v,qdim=2048,kdim=256,hd=256,offset=offset*256),'rope',{'qkv':qkv,'rope':rope,'q':q,'k':k,'v':v},qdim=2048,kdim=256,hd=256,offset=offset*256)
        if last and prefix:return
        # Prefix attention must not read the suffix cache. Its pad key is
        # explicitly zeroed each replay because suffix inference overwrites it.
        if prefix and nk_pad>nk:
            self.zero(k[nk:nk_pad]);self.zero(v[nk:nk_pad])
        qr=q.view(rows*8,256);ao=attout.view(rows*8,256)
        self.mm(qr,k[:nk_pad],logits,alpha=1./16.)
        self.op(lambda:self.support.softmax(logits,mask_rows=0 if prefix else 8,mask_start=nk if prefix else offset+1,pad_start=nk),'softmax',{'x':logits},mask_rows=0 if prefix else 8,mask_start=nk if prefix else offset+1,pad_start=nk)
        self.mm(logits,v[:nk_pad].T,ao)
        ow=self.weight((tag,index,'o'),att.o_proj.weight)
        if prefix:self.cutlass(attout,ow,x,1,beta=1.)
        else:self.mm(attout,ow,x,beta=1.)
        self.norm(x,layer.post_attention_layernorm,norm,rms=True,folded=True)
        mlp=layer.mlp;hidden_dim=mlp.gate_proj.weight.shape[0]
        hidden=self.buffer(tag+'.hidden',(rows,hidden_dim))
        scale=1.+layer.post_attention_layernorm.weight.float()
        gw=self.weight((tag,index,'gate'),mlp.gate_proj.weight.float()*scale)
        uw=self.weight((tag,index,'up'),mlp.up_proj.weight.float()*scale)
        if prefix:
            fused=self.owners.enter_context(ShadowGeGLU(norm,gw,uw,output=hidden));self.op(fused,'geglu',{'x':norm,'gate':gw,'up':uw,'out':hidden})
        else:
            gate=self.buffer(tag+'.gate',(rows,hidden_dim));up=self.buffer(tag+'.up',(rows,hidden_dim))
            self.mm(norm,gw,gate);self.mm(norm,uw,up)
            self.op(lambda:self.support.gate(gate,up,hidden),'gate',{'gate':gate,'up':up,'out':hidden})
        dw=self.weight((tag,index,'down'),mlp.down_proj.weight)
        if prefix:self.cutlass(hidden,dw,x,2,beta=1.)
        else:self.mm(hidden,dw,x,beta=1.)

    def build(self):
        pg=self.model.paligemma_with_expert;vision=pg.paligemma.vision_tower.vision_model
        m=self.views*256
        patches=self.buffer('patches',(m,588));x=self.buffer('vision.x',(m,1152));norm=self.buffer('vision.norm',(m,1152))
        emb=vision.embeddings;conv=emb.patch_embedding
        pw=self.weight('patch.weight',conv.weight.permute(0,2,3,1).reshape(1152,588));pb=self.weight('patch.bias',conv.bias)
        pos=self.weight('patch.pos',emb.position_embedding.weight)
        self.op(lambda:self.support.patch(self.images,patches),'patch',{'x':self.images,'out':patches});self.mm(patches,pw,x)
        self.op(lambda:self.support.patch_bias(x,pb,pos),'patch_bias',{'x':x,'bias':pb,'pos':pos})
        qkv=self.buffer('vision.qkv',(m,3456));ao=self.buffer('vision.attout',(m,1152))
        projected=self.buffer('vision.projected',(m,1152));hidden=self.buffer('vision.hidden',(m,4304))
        for i,layer in enumerate(vision.encoder.layers):
            self.norm(x,layer.layer_norm1,norm)
            att=layer.self_attn;qw=self.weight(('vqkv',i),torch.cat([p.weight for p in (att.q_proj,att.k_proj,att.v_proj)]))
            qb=self.weight(('vqkvbias',i),torch.cat([p.bias for p in (att.q_proj,att.k_proj,att.v_proj)]))
            self.mm(norm,qw,qkv);self.bias(qkv,qb)
            self.op(lambda:self.fmha(qkv,ao,batches=self.views),'fmha',{'qkv':qkv,'out':ao},batches=self.views,rows=256,heads=16,hd=72)
            ow=self.weight(('vo',i),att.out_proj.weight);ob=self.weight(('vobias',i),att.out_proj.bias)
            self.mm(ao,ow,projected)
            self.op(lambda projected=projected,ob=ob,x=x:self.support.bias_residual(projected,ob,x),'bias_residual',{'x':projected,'bias':ob,'residual':x});self.copy(x,projected)
            self.norm(x,layer.layer_norm2,norm)
            fc1=layer.mlp.fc1;fc2=layer.mlp.fc2
            w1=self.weight(('vfc1',i),fc1.weight);b1=self.weight(('vfc1bias',i),fc1.bias)
            w2=self.weight(('vfc2',i),fc2.weight);b2=self.weight(('vfc2bias',i),fc2.bias)
            self.mm(norm,w1,hidden);self.bias(hidden,b1);self.op(lambda:self.support.gelu(hidden),'gelu',{'x':hidden});self.mm(hidden,w2,projected)
            self.op(lambda b2=b2:self.support.bias_residual(projected,b2,x),'bias_residual',{'x':projected,'bias':b2,'residual':x});self.copy(x,projected)
        self.norm(x,vision.post_layernorm,norm)
        projector=pg.paligemma.multi_modal_projector.linear
        projected_image=self.buffer('vision.tokens',(m,2048))
        self.mm(norm,self.weight('projector',projector.weight),projected_image);self.bias(projected_image,self.weight('projectorbias',projector.bias))
        self.components['vision']=(0,len(self.ops));prefix_start=len(self.ops)
        seq=m+self.prompt;px=self.buffer('prefix.x',(seq,2048))
        self.copy(px[:m],projected_image)
        if self.prompt:
            embedding=pg.paligemma.language_model.embed_tokens.weight
            self.op(lambda:px[m:].copy_((embedding[self.tokens].float()*math.sqrt(2048.)).bfloat16()),'embedding',{'weight':embedding,'tokens':self.tokens,'out':px[m:]},scale=math.sqrt(2048.))
        lm=pg.paligemma.language_model
        theta=getattr(lm.config,'rope_theta',10000.)
        prope=self.rope(seq,0,256,theta);self.buffers['prefix.rope']=prope
        total=seq+self.chunk+1;cache_rows=total+1
        caches=[(self.buffer(f'cache.k.{i}',(cache_rows,256)),self.buffer(f'cache.v.{i}',(cache_rows,256))) for i in range(len(lm.layers))]
        for i,layer in enumerate(lm.layers):self.gemma(px,layer,i,prope,caches[i],prefix=True,last=i==len(lm.layers)-1)
        self.components['prefix']=(prefix_start,len(self.ops));expert_start=len(self.ops)
        state_token=self.buffer('state.token',(1,1024));self.mm(self.state,self.weight('state.w',self.model.state_proj.weight),state_token);self.bias(state_token,self.weight('state.b',self.model.state_proj.bias))
        sx=self.buffer('expert.x',(self.chunk+1,1024));action=sx[1:];temp=self.buffer('action.temp',(self.chunk,1024))
        ain=self.model.action_in_proj;ti=self.model.action_time_mlp_in;to=self.model.action_time_mlp_out
        aw=self.weight('action.in.w',ain.weight);ab=self.weight('action.in.b',ain.bias)
        wa=self.weight('action.time.action',ti.weight[:,:1024]);wt=self.weight('action.time.time',ti.weight[:,1024:]);tb=self.weight('action.time.bias',ti.bias)
        tw=self.weight('action.time.out.w',to.weight);tob=self.weight('action.time.out.b',to.bias)
        expert=pg.gemma_expert.model;srope=self.rope(self.chunk+1,seq,256,getattr(expert.config,'rope_theta',10000.));self.buffers['expert.rope']=srope
        final=self.buffer('expert.final',(self.chunk+1,1024));out=self.model.action_out_proj
        # Match source BF16 setup multiply, after casting the source weight.
        ow=self.weight('action.out.w',out.weight).mul(-1./self.steps);ob=self.weight('action.out.b',out.bias).mul(-1./self.steps)
        self.weights['action.out.scaled.w']=ow;self.weights['action.out.scaled.b']=ob
        for step in range(self.steps):
            fraction=torch.linspace(0,1,512,device=self.device,dtype=torch.float64)
            angle=(1./(.004*(4./.004)**fraction)*2*math.pi)*float(torch.tensor(1.-step/self.steps,dtype=torch.float32))
            te=torch.cat((angle.sin(),angle.cos())).bfloat16()
            time_bias=((te@wt.T)+tb).expand(self.chunk,-1).contiguous();self.buffers[f'time.bias.{step}']=time_bias
            self.copy(sx[:1],state_token);self.mm(self.noise,aw,action);self.bias(action,ab)
            self.mm(action,wa,temp);self.op(lambda time_bias=time_bias:self.support.add_silu(temp,time_bias),'add_silu',{'x':temp,'bias':time_bias})
            self.mm(temp,tw,action);self.bias(action,tob)
            for i,layer in enumerate(expert.layers):self.gemma(sx,layer,i,srope,caches[i])
            self.norm(sx,expert.norm,final,rms=True)
            self.mm(final[1:],ow,self.noise,beta=1.);self.bias(self.noise,ob)
        self.components['expert']=(expert_start,len(self.ops));self.output=self.noise.view(1,self.chunk,32)

    @torch.no_grad()
    def run(self):
        for op in self.ops:op()
        return self.output
    @torch.no_grad()
    def forward(self,images,state,noise,tokens):
        self.images.copy_(images);self.state.copy_(state.reshape_as(self.state));self.noise.copy_(noise.reshape_as(self.noise));self.tokens.copy_(tokens.reshape_as(self.tokens))
        if self.graph is None:self.run()
        else:self.graph.replay()
        return self.output
    def capture(self):
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.run();self.run()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.run()
    def close(self):self.graph=None;self.owners.close()
