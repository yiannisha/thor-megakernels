"""Execute the full native decoder once under a selected CUDA sanitizer."""
import json
import torch
from tools.thor.model_bench import ModelAdapter
with torch.inference_mode():
 a=ModelAdapter('native',1024,1,cache_capacity=1125);a.reset();out=a.run();torch.cuda.synchronize();assert bool(torch.isfinite(out).all())
 print(json.dumps({'status':'PASS','scope':'full native decoder','resources':a.metadata['native_resources'],'build':a.metadata['native_build']}))
