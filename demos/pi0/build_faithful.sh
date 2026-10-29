#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/thor/env.sh
mkdir -p local/pi0/faithful
python demos/pi0/faithful_pi0/prepare.py
PI0_FAITHFUL_CUTLASS="$MEGAKERNELS_ROOT/local/pi0/faithful/cutlass"
PI0_FAITHFUL_FLAGS=(-O3 --use_fast_math -lineinfo -std=c++20 -gencode=arch=compute_110a,code=sm_110a
 --expt-relaxed-constexpr --expt-extended-lambda -Xcompiler=-fPIC
 -Ilocal/pi0/faithful/sources -I"$PI0_FAITHFUL_CUTLASS/include" -I"$PI0_FAITHFUL_CUTLASS/tools/util/include")
"$NVCC" "${PI0_FAITHFUL_FLAGS[@]}" -shared demos/pi0/faithful_pi0/shadow_gemm.cu \
 -o local/pi0/faithful/shadow_gemm.so -lcuda
PI0_FLASHRT_ROOT="$MEGAKERNELS_ROOT/../FlashRT/csrc"
"$NVCC" "${PI0_FAITHFUL_FLAGS[@]}" -I"$PI0_FLASHRT_ROOT/kernels" -shared \
 demos/pi0/faithful_pi0/shadow_support.cu "$PI0_FLASHRT_ROOT/kernels/norm.cu" "$PI0_FLASHRT_ROOT/kernels/activation.cu" \
 -o local/pi0/faithful/shadow_support.so
# The source GeGLU overlay must precede the pinned stock CUTLASS headers.
"$NVCC" -I"$PI0_FLASHRT_ROOT/gemm/mega" "${PI0_FAITHFUL_FLAGS[@]}" -I"$PI0_FLASHRT_ROOT/kernels" -shared \
 demos/pi0/faithful_pi0/shadow_geglu.cu -Llocal/pi0/faithful -l:shadow_gemm.so -l:shadow_support.so \
 -Xlinker=-rpath -Xlinker='$ORIGIN' -o local/pi0/faithful/shadow_geglu.so
"$NVCC" "${PI0_FAITHFUL_FLAGS[@]}" -I"$PI0_FLASHRT_ROOT/attention" -I"$PI0_FAITHFUL_CUTLASS/examples/77_blackwell_fmha" -shared \
 local/pi0/faithful/sources/fmha_bf16.cu -o local/pi0/faithful/shadow_fmha.so
"$NVCC" -Ilocal/pi0/faithful/sources/vm -I"$PI0_FLASHRT_ROOT/gemm/mega" "${PI0_FAITHFUL_FLAGS[@]}" -I"$PI0_FLASHRT_ROOT/kernels" -Iinclude \
 -I"$THUNDERKITTENS_ROOT/include" -Idemos/pi0/faithful_pi0 \
 -DMK_THOR -DKITTENS_HOPPER -DMK_THOR_REGISTER_CAP=255 -shared \
 demos/pi0/faithful_pi0/native_vm.cu -o local/pi0/faithful/native_program.so
# Group 1 is a separate VM kernel: PTX forbids mixing TCGEN CTA groups.
"$NVCC" -Ilocal/pi0/faithful/sources/vm "${PI0_FAITHFUL_FLAGS[@]}" -Iinclude \
 -I"$THUNDERKITTENS_ROOT/include" -Idemos/pi0/faithful_pi0 \
 -I"$PI0_FLASHRT_ROOT/attention" -I"$PI0_FAITHFUL_CUTLASS/examples/77_blackwell_fmha" \
 -DMK_THOR -DKITTENS_HOPPER -DMK_THOR_REGISTER_CAP=128 -shared \
 demos/pi0/faithful_pi0/native_fmha_vm.cu -o local/pi0/faithful/native_fmha_vm.so
