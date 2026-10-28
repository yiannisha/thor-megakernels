#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/thor/env.sh
"$NVCC" -std=c++20 -O3 -lineinfo -gencode=arch=compute_110a,code=sm_110a --expt-relaxed-constexpr --expt-extended-lambda \
 -DKITTENS_HOPPER -DMK_THOR -Xcompiler=-fPIC -shared -Xptxas=-v \
 -I"$THUNDERKITTENS_ROOT/include" -Iinclude -Ilocal/cutlass-nvfp4/include \
 -Ilocal/cutlass-nvfp4/tools/util/include $(python -m pybind11 --includes) \
 experimental/native_fp4_vm/model.cu -o "local/thor-native-fp4-vm/mk_fp4_model$(python3-config --extension-suffix)" -lcuda
python - <<'PY'
import hashlib,json,sysconfig
from pathlib import Path
from tools.thor.repro import source_hash
p=Path('local/thor-native-fp4-vm')/('mk_fp4_model'+sysconfig.get_config_var('EXT_SUFFIX'))
d=dict(binary_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),framework_source_hash=source_hash(),
 source_sha256={'tools/analysis/nvfp4/primitive.cu':hashlib.sha256(Path('tools/analysis/nvfp4/primitive.cu').read_bytes()).hexdigest()},
 model_sources={str(s):hashlib.sha256(s.read_bytes()).hexdigest() for s in Path('experimental/native_fp4_vm').glob('*.cu')})
Path(str(p)+'.json').write_text(json.dumps(d,indent=2)+'\n')
PY
