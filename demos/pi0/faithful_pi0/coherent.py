"""One- and three-launch VM experiments with compatible tensor-core operators."""
import ctypes as C
import os
import torch
from .program import Operation
from .native import NativeLibrary,NativeModel,ROOT

class CoherentModel(NativeModel):
    def __init__(self,shadow,*,partition='single',library=None,geglu='two-phase',down='generic',action_library=None):
        if geglu not in ('fused','two-phase') or down not in ('generic','source'):raise ValueError((geglu,down))
        if partition not in ('single','three'):raise ValueError(partition)
        self.shadow=shadow;self.partition=partition;self.api=NativeLibrary(library or ROOT/'local/pi0/faithful/coherent_program.so')
        self.trace=False;self.parts=[];self.part_apis=[];self.regions=[];self.region_apis=[];self.region_ranges=[];self.graph=None
        self.action_api=NativeLibrary(action_library) if partition=="three" and action_library else self.api
        self.api.lib.faithful_vm_vision_create.argtypes=[C.POINTER(C.c_void_p)]+[C.c_void_p]*2+[C.c_int]*2+[C.c_void_p]
        self.api.lib.faithful_vm_vision_create.restype=C.c_int
        self.api.lib.faithful_vm_coherent_geglu_create.argtypes=[C.POINTER(C.c_void_p)]+[C.c_void_p]*4+[C.c_int]*3+[C.c_void_p]
        self.api.lib.faithful_vm_coherent_geglu_create.restype=C.c_int
        starts={a for a,b in shadow.components.values()} if partition=='three' else set()
        self.marker=None;self.marker_values=[]
        if os.environ.get('PI0_COHERENT_MARKERS')=='1':self.marker=torch.full((1,2),-1.,dtype=torch.bfloat16,pin_memory=True)
        region=[];indices=[];active_api=self.api
        def flush():
            if not region:return
            raw=(C.c_void_p*len(region))(*(h.value for h in region))
            h=active_api.create('merge',raw,len(region))
            self.regions.append(('group2',h));self.region_apis.append(active_api);self.region_ranges.append((indices[0],indices[-1]));region.clear();indices.clear()
        try:
            for i,op in enumerate(shadow.ops):
                if i in starts:
                    flush()
                    if i==shadow.components["expert"][0]:active_api=self.action_api
                if op.kind=='fmha':
                    p=op.params;t=op.tensors
                    if p['heads']!=16 or p['hd']!=72:raise ValueError('SigLIP geometry')
                    h=active_api.create('vision_create',t['qkv'].data_ptr(),t['out'].data_ptr(),p['batches'],p['rows'])
                elif op.kind=='gemm' and op.params.get('tactic')==2 and down=='generic':
                    t=op.tensors;p=op.params
                    h=active_api.generic(t['x'],t['weight'],t['out'],p['alpha'],p['beta'])
                elif op.kind=='geglu' and geglu=='two-phase':
                    t=op.tensors;x,g,u,out=(t[k] for k in ('x','gate','up','out'))
                    h=active_api.create('coherent_geglu_create',x.data_ptr(),g.data_ptr(),u.data_ptr(),out.data_ptr(),x.shape[0],g.shape[0],x.shape[1])
                else:h=active_api.operation(op)
                self.parts.append(h);self.part_apis.append(active_api);region.append(h);indices.append(i)
                if self.marker is not None:
                    value=torch.tensor([[i//256,i%256]],device=shadow.device,dtype=torch.bfloat16);self.marker_values.append(value)
                    mark=active_api.operation(Operation(lambda:None,'copy',{'x':value,'out':self.marker}))
                    self.parts.append(mark);self.part_apis.append(active_api);region.append(mark);indices.append(i)
            flush()
        except Exception:self.close();raise
        expected=1 if partition=='single' else 3
        if len(self.regions)!=expected:raise AssertionError((partition,len(self.regions)))
        self.output=shadow.output

    def run(self):
        for index,((_,h),api) in enumerate(zip(self.regions,self.region_apis)):
            if self.trace:print('REGION START',index,self.region_ranges[index],flush=True)
            api.run(h)
            if self.trace:torch.cuda.synchronize();print('REGION PASS',index,flush=True)
        return self.output

    def close(self):
        self.graph=None
        for (_,h),api in reversed(list(zip(self.regions,self.region_apis))):api.lib.faithful_vm_destroy(h)
        self.regions.clear();self.region_apis.clear()
        for h,api in reversed(list(zip(self.parts,self.part_apis))):api.lib.faithful_vm_destroy(h)
        self.parts.clear();self.part_apis.clear()
