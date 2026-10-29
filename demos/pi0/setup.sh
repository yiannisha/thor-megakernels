#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/thor/env.sh
mkdir -p local/pi0
python -m venv local/pi0/venv
PI0_TORCH_SITE=$(python -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).parent.parent)')
local/pi0/venv/bin/python - "$PI0_TORCH_SITE" <<'PY'
import site,sys
from pathlib import Path
Path(site.getsitepackages()[0],'thor.pth').write_text(sys.argv[1]+'\n')
PY
local/pi0/venv/bin/python -m pip install transformers==4.53.2 flax==0.10.2 jax==0.5.3 \
 orbax-checkpoint==0.11.13 jaxtyping==0.2.36 beartype==0.19.0 augmax==0.4.1 sentencepiece==0.2.2
if [[ ! -d local/pi0/openpi/.git ]]; then
 git clone --filter=blob:none --no-checkout https://github.com/Physical-Intelligence/openpi.git local/pi0/openpi
fi
git -C local/pi0/openpi checkout 215abfb217dbac7d5f1273282331b9b1866c0479
local/pi0/venv/bin/python - <<'PY'
from pathlib import Path
import shutil,transformers
assert transformers.__version__=='4.53.2'
shutil.copytree(Path('local/pi0/openpi/src/openpi/models_pytorch/transformers_replace'),
               Path(transformers.__file__).parent,dirs_exist_ok=True)
PY
python demos/pi0/prepare_fa2.py --fetch
