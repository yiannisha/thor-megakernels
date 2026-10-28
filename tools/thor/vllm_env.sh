# Source from bash. Uses a separate vLLM environment and its own pip CUDA runtime.
THOR_VLLM_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export CUDA_HOME="$THOR_VLLM_ROOT/local/toolchain/usr/local/cuda-13.2"
export PATH="$THOR_VLLM_ROOT/local/vllm-venv/bin:$THOR_VLLM_ROOT/local/bin:$CUDA_HOME/bin:$PATH"
export CXX="$THOR_VLLM_ROOT/local/bin/g++"
export NVCC_CCBIN="$CXX"
export CPATH="$THOR_VLLM_ROOT/local/vllm-include:$THOR_VLLM_ROOT/local/toolchain/usr/include/python3.12:$THOR_VLLM_ROOT/local/toolchain/usr/include:$THOR_VLLM_ROOT/local/toolchain/usr/include/aarch64-linux-gnu${CPATH:+:$CPATH}"
export VLLM_NO_USAGE_STATS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DO_NOT_TRACK=1
