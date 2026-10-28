# Source this file from bash to use the project-local JetPack toolkit and venv.
THOR_PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export CUDA_HOME="$THOR_PROJECT_ROOT/local/toolchain/usr/local/cuda-13.2"
export PATH="$THOR_PROJECT_ROOT/local/toolchain/opt/nvidia/nsight-systems/2025.6.3/bin:$THOR_PROJECT_ROOT/local/toolchain/opt/nvidia/nsight-compute/2026.1.1:$THOR_PROJECT_ROOT/local/thor-venv/bin:$THOR_PROJECT_ROOT/local/bin:$CUDA_HOME/bin:$PATH"
export CXX="$THOR_PROJECT_ROOT/local/bin/g++"
export NVCC="$CUDA_HOME/bin/nvcc"
export NVCC_CCBIN="$CXX"
export CPATH="$THOR_PROJECT_ROOT/local/toolchain/usr/include/python3.12:$THOR_PROJECT_ROOT/local/toolchain/usr/include:$THOR_PROJECT_ROOT/local/toolchain/usr/include/aarch64-linux-gnu${CPATH:+:$CPATH}"
export LIBRARY_PATH="$THOR_PROJECT_ROOT/local/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export THUNDERKITTENS_ROOT="$THOR_PROJECT_ROOT/local/thunderkittens"
export MEGAKERNELS_ROOT="$THOR_PROJECT_ROOT"
