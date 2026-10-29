#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/thor/env.sh
if [[ ${PI0_COHERENT_SKIP_PREPARE:-0} != 1 ]]; then
 python demos/pi0/faithful_pi0/prepare.py
 python demos/pi0/prepare_fa2.py
fi
PI0_COHERENT_TORCH_INCLUDE=$(python -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).parent/"include")')
PI0_COHERENT_OUTPUT=${PI0_COHERENT_OUTPUT:-local/pi0/faithful/coherent_program.so}
PI0_COHERENT_EXTRA=(-DPI0_GENERIC_M=${PI0_GENERIC_M:-128} -DPI0_GENERIC_N=${PI0_GENERIC_N:-128})
if [[ ${PI0_COHERENT_FAST_BARRIERS:-0} == 1 ]]; then PI0_COHERENT_EXTRA+=(-DPI0_COHERENT_FAST_BARRIERS); fi
case ${PI0_COHERENT_PHASE:-all} in
 action) PI0_COHERENT_EXTRA+=(-DPI0_PHASE_ACTION);;
 vision) PI0_COHERENT_EXTRA+=(-DPI0_PHASE_VISION);;
 vlm) PI0_COHERENT_EXTRA+=(-DPI0_PHASE_VLM);;
 all) ;;
 *) exit 2;;
esac
"$NVCC" "${PI0_COHERENT_EXTRA[@]}" -O3 --use_fast_math -lineinfo -std=c++20 -gencode=arch=compute_110a,code=sm_110a \
 --expt-relaxed-constexpr --expt-extended-lambda -Xcompiler=-fPIC -Xptxas=-v \
 -DMK_THOR -DKITTENS_HOPPER -DMK_THOR_REGISTER_CAP=${PI0_COHERENT_REGISTERS:-255} -DPI0_UNIFIED \
 -Ilocal/pi0/faithful/sources/vm -I../FlashRT/csrc/gemm/mega -I../FlashRT/csrc/kernels \
 -Ilocal/pi0/faithful/sources -Iinclude -I"$THUNDERKITTENS_ROOT/include" \
 -Idemos/pi0/faithful_pi0 -Ilocal/pi0/faithful/cutlass/include \
 -Ilocal/pi0/faithful/cutlass/tools/util/include -Ilocal/pi0/flash-attention-overlay \
 -I"$PI0_COHERENT_TORCH_INCLUDE" -shared demos/pi0/faithful_pi0/native_vm.cu \
 -o "$PI0_COHERENT_OUTPUT"
