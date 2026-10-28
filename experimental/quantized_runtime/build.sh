#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/thor/env.sh
bash tools/analysis/nvfp4/setup_cutlass.sh
python -m tools.analysis.nvfp4.cutlass_overlay
variant="${1:-auto}"
stages=0
if [[ "$variant" != auto ]]; then stages="$variant"; fi
"$NVCC" -std=c++17 -O3 -lineinfo -arch=sm_110a --expt-relaxed-constexpr \
 -DNV_STAGES="$stages" -Xcompiler=-fPIC -shared \
 -Ilocal/thor-nvfp4/overlay -Ilocal/cutlass-nvfp4/include -Ilocal/cutlass-nvfp4/tools/util/include \
 experimental/quantized_runtime/persistent.cu \
 -o "local/thor-nvfp4/persistent-$variant.so" -lcuda
python -m tools.analysis.nvfp4.build_metadata "local/thor-nvfp4/persistent-$variant.so" "$variant"
