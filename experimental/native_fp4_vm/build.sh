#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/thor/env.sh
bash tools/analysis/nvfp4/setup_cutlass.sh
"$NVCC" -std=c++20 -O3 -lineinfo -gencode=arch=compute_110a,code=sm_110a --expt-relaxed-constexpr --expt-extended-lambda \
 -DKITTENS_HOPPER -DMK_THOR -Xcompiler=-fPIC -shared -Xptxas=-v \
 -I"$THUNDERKITTENS_ROOT/include" -Iinclude -Ilocal/cutlass-nvfp4/include \
 -Ilocal/cutlass-nvfp4/tools/util/include experimental/native_fp4_vm/vm.cu \
 -o local/thor-native-fp4-vm/vm.so -lcuda
python -m tools.analysis.nvfp4.build_metadata local/thor-native-fp4-vm/vm.so 2
python - <<'PY'
import json,hashlib
from pathlib import Path
p=Path('local/thor-native-fp4-vm/vm.so.json');d=json.loads(p.read_text())
names=['include/megakernel.cuh','include/util.cuh','include/ops/nvfp4_linear.cuh','include/ops/nvfp4_types.cuh','experimental/native_fp4_vm/vm.cu']
d['vm_sources']={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in names}
from tools.thor.repro import source_hash
d['framework_source_hash']=source_hash()
p.write_text(json.dumps(d,indent=2)+'\n')
PY
