#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/thor/env.sh
bash tools/analysis/nvfp4/setup_cutlass.sh
"$NVCC" -std=c++20 -O3 -lineinfo -gencode=arch=compute_110a,code=sm_110a --expt-relaxed-constexpr \
 -Xcompiler=-fPIC -shared -Xptxas=-v -Ilocal/cutlass-nvfp4/include -Ilocal/cutlass-nvfp4/tools/util/include \
 experimental/weight_only_fp4/primitive.cu -o local/thor-w4a16/primitive.so -lcuda
python -m tools.analysis.nvfp4.build_metadata local/thor-w4a16/primitive.so

"$NVCC" -std=c++17 -O2 -shared -Xcompiler=-fPIC \
 experimental/weight_only_fp4/cache_policy.cu -o local/thor-w4a16/cache_policy.so
python - <<'META'
import hashlib,json
from pathlib import Path
p=Path('local/thor-w4a16/primitive.so.json');d=json.loads(p.read_text())
for name in ('experimental/weight_only_fp4/primitive.cu','experimental/weight_only_fp4/cache_policy.cu'):
 d['source_sha256'][name]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
d['cache_policy_binary_sha256']=hashlib.sha256(Path('local/thor-w4a16/cache_policy.so').read_bytes()).hexdigest()
p.write_text(json.dumps(d,indent=2)+'\n')
META
