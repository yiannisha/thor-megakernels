"""Install the pinned ARM64 vLLM baseline separately from the native port."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headers-only', action='store_true')
    args = parser.parse_args()
    env = ROOT / 'local/vllm-venv'
    if not args.headers_only:
        if not env.exists():
            subprocess.run([sys.executable, '-m', 'venv', str(env)], check=True)
        subprocess.run([str(env / 'bin/python'), '-m', 'pip', 'install',
                        '-c', str(ROOT / 'docs/thor/vllm-constraints.txt'),
                        '-r', str(ROOT / 'docs/thor/vllm-python-lock.txt')], check=True)
        subprocess.run([str(env / 'bin/python'), '-m', 'pip', 'check'], check=True)
    # Expose cuRAND only. Exposing the entire pip CUDA 13.0 include tree would
    # override the local 13.2 runtime headers and break CCCL/compiler matching.
    include = ROOT / 'local/vllm-include'
    include.mkdir(exist_ok=True)
    matches = list((env / 'lib/python3.12/site-packages/nvidia/cu13/include').glob('curand*'))
    if not matches:
        raise RuntimeError('installed cuRAND headers unavailable')
    for header in matches:
        link = include / header.name
        if link.is_symlink():
            link.unlink()
        link.symlink_to(header)


if __name__ == '__main__':
    main()
