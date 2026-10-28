#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
source tools/thor/env.sh
bash tools/analysis/nvfp4/setup_cutlass.sh
mkdir -p local/thor-nvfp4
"$NVCC" -std=c++17 -O3 -arch=sm_110a --expt-relaxed-constexpr \
 -Xcompiler=-fPIC -shared -Ilocal/cutlass-nvfp4/include \
 -Ilocal/cutlass-nvfp4/tools/util/include tools/analysis/nvfp4/primitive.cu \
 -o local/thor-nvfp4/primitive.so -lcuda
python -m tools.analysis.nvfp4.build_metadata local/thor-nvfp4/primitive.so
