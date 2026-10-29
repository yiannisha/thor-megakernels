# Source from bash after running the local PiZero setup.
PI0_DEMO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$PI0_DEMO_ROOT/../../tools/thor/env.sh"
source "$MEGAKERNELS_ROOT/local/pi0/venv/bin/activate"
export OPENPI_ROOT="$MEGAKERNELS_ROOT/local/pi0/openpi"
export PYTHONPATH="$PI0_DEMO_ROOT:$OPENPI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu
