#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/thor/env.sh
mkdir -p local/pi0
PI0_REGISTERS=${PI0_REGISTERS:-255}
PI0_LINEAR_STAGES=${PI0_LINEAR_STAGES:-2}
PI0_VISION_KEY_TILE=${PI0_VISION_KEY_TILE:-64}
PI0_WEIGHT_LOCALITY=${PI0_WEIGHT_LOCALITY:-0}
PI0_OUTPUT=${PI0_OUTPUT:-local/pi0/model.so}
python -m tools.thor.prepare_tk
python demos/pi0/prepare_fa2.py
PI0_TORCH_INCLUDE=$(python -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).parent/"include")')
PI0_CUPTI_ROOT=$(python -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).parent.parent/"nvidia/cu13")')
python - "$PI0_OUTPUT.inputs.json" <<'PY'
import hashlib,json,subprocess,torch
from pathlib import Path
files=list(Path('demos/pi0/native_pi0').glob('*'))+list(Path('include').rglob('*.cuh'))
files += [Path('demos/pi0/build.sh'),Path('demos/pi0/prepare_fa2.py')]
snapshot={'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()},
          'compiler':subprocess.check_output(['nvcc','--version'],text=True),
          'tk_overlay':json.loads(Path('local/thunderkittens/overlay.json').read_text()),
          'fa2_overlay':json.loads(Path('local/pi0/flash-attention-overlay/manifest.json').read_text()),
          'cutlass_revision':subprocess.check_output(['git','-C','local/cutlass-nvfp4','rev-parse','HEAD'],text=True).strip(),
          'torch_version':torch.__version__,'torch_revision':torch.version.git_version}
Path(__import__('sys').argv[1]).write_text(json.dumps(snapshot,indent=2)+'\n')
PY
"$NVCC" -O3 -lineinfo -std=c++20 -gencode=arch=compute_110a,code=sm_110a \
 -DKITTENS_HOPPER -DKITTENS_BLACKWELL -DMK_THOR -DNATIVE_COMPUTE_CONSUMER_WARPS=4 \
 -DMK_THOR_REGISTER_CAP="$PI0_REGISTERS" -DPI0_LINEAR_STAGES="$PI0_LINEAR_STAGES" \
 -DPI0_VISION_KEY_TILE="$PI0_VISION_KEY_TILE" -DPI0_WEIGHT_LOCALITY="$PI0_WEIGHT_LOCALITY" \
 --expt-relaxed-constexpr --expt-extended-lambda -Iinclude -I"$THUNDERKITTENS_ROOT/include" \
 -I"$PI0_CUPTI_ROOT/include" -I"$PI0_TORCH_INCLUDE" \
 -Ilocal/pi0/flash-attention-overlay -Ilocal/cutlass-nvfp4/include -shared -Xcompiler -fPIC -Xptxas=-v \
 demos/pi0/native_pi0/model.cu demos/pi0/native_pi0/launch_trace.cpp \
 -L"$PI0_CUPTI_ROOT/lib" -Xlinker -rpath -Xlinker "$PI0_CUPTI_ROOT/lib" -l:libcupti.so.13 -lcuda \
 -o "$PI0_OUTPUT"
python - "$PI0_OUTPUT" "$PI0_REGISTERS" "$PI0_LINEAR_STAGES" "$PI0_VISION_KEY_TILE" "$PI0_WEIGHT_LOCALITY" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path
snapshot=json.loads(Path(sys.argv[1]+'.inputs.json').read_text())
for name,digest in snapshot['sources'].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
        raise RuntimeError('Build inputs changed during compilation: '+name)
for name,digest in snapshot['fa2_overlay']['headers'].items():
    if hashlib.sha256(Path('local/pi0/flash-attention-overlay',name).read_bytes()).hexdigest()!=digest:
        raise RuntimeError('FlashAttention inputs changed during compilation: '+name)
manifest={'arch':'sm_110a','workers':20,'consumer_warps':4,'register_cap':int(sys.argv[2]),
 'linear_stages':int(sys.argv[3]),'vision_key_tile':int(sys.argv[4]),'weight_locality':bool(int(sys.argv[5])),
 'runtime_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 **snapshot,
 'binary_sha256':hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()}
Path(sys.argv[1]+'.json').write_text(json.dumps(manifest,indent=2)+'\n')
PY
