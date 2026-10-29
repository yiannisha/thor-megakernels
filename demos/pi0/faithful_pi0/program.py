"""Source-policy operations shared by standalone and native program builders."""
from dataclasses import dataclass,field
from typing import Callable
import torch

@dataclass
class Operation:
    run:Callable
    kind:str
    tensors:dict=field(default_factory=dict)
    params:dict=field(default_factory=dict)
    def __call__(self):return self.run()
    def manifest(self):
        return {'kind':self.kind,'tensors':{name:{'shape':list(t.shape),'stride':list(t.stride()),'dtype':str(t.dtype),'storage_offset':t.storage_offset()} for name,t in self.tensors.items()},'params':self.params}
