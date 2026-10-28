#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/thor/env.sh
bash tools/analysis/nvfp4/setup_cutlass.sh
"$NVCC" -std=c++20 -O3 -lineinfo -gencode=arch=compute_110a,code=sm_110a --expt-relaxed-constexpr \
 -Xcompiler=-fPIC -shared -Xptxas=-v -Ilocal/cutlass-nvfp4/include -Ilocal/cutlass-nvfp4/tools/util/include \
 experimental/weight_only_fp4/primitive.cu -o local/thor-w4a16/primitive.so -lcuda
python -m tools.analysis.nvfp4.build_metadata local/thor-w4a16/primitive.so
