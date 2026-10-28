#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
revision=91843b71d90737e0a9b86f7d9df213cff5a3526f
checkout=local/cutlass-nvfp4
if [[ ! -d "$checkout/.git" ]]; then
 git clone --filter=blob:none --no-checkout https://github.com/NVIDIA/cutlass.git "$checkout"
 git -C "$checkout" checkout "$revision"
fi
if [[ "$(git -C "$checkout" rev-parse HEAD)" != "$revision" ]]; then
 echo "Expected CUTLASS revision $revision in $checkout" >&2
 exit 1
fi
if [[ -n "$(git -C "$checkout" status --porcelain)" ]]; then
 echo 'CUTLASS checkout has modifications; use the generated overlay instead.' >&2
 exit 1
fi
