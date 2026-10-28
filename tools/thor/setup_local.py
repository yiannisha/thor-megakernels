"""Reproduce the project-local environment from recorded official packages.

No sudo, dpkg installation, driver changes or system-Python writes. APT downloads
use the operator's configured signed repositories; hashes pin the captured files.
Requires this project's Ubuntu 24.04 ARM64 / JetPack r39.2 base system.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def run(args, **kwargs):
    subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def install():
    if platform.machine() != 'aarch64':
        raise RuntimeError('this recorded setup is for ARM64 only')
    packages = json.loads((ROOT / 'docs/thor/apt-lock.json').read_text())
    download = ROOT / 'local/toolchain-debs'
    prefix = ROOT / 'local/toolchain'
    download.mkdir(parents=True, exist_ok=True)
    prefix.mkdir(parents=True, exist_ok=True)
    for package in packages:
        path = download / package['filename']
        if not path.exists():
            run(['apt-get', 'download', package['package'] + '=' + package['version']], cwd=download)
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != package['sha256']:
            raise RuntimeError(f'package hash mismatch: {path}; refusing extraction')
        run(['dpkg-deb', '-x', path, prefix])
    scripts = ROOT / 'local/bin'
    scripts.mkdir(exist_ok=True)
    wrappers = {
        'g++': 'exec "$prefix/toolchain/usr/bin/aarch64-linux-gnu-g++-13" -B/usr/lib/gcc/aarch64-linux-gnu/13/ "$@"',
        'python3-config': 'exec "$prefix/toolchain/usr/bin/aarch64-linux-gnu-python3.12-config" "$@"'}
    for name, body in wrappers.items():
        target = scripts / name
        target.write_text('#!/bin/bash\nprefix=$(cd -- "$(dirname -- "$0")/.." && pwd)\n' + body + '\n')
        target.chmod(0o755)
    cxx = scripts / 'c++'
    if not cxx.exists():
        cxx.symlink_to('g++')
    venv = ROOT / 'local/thor-venv'
    if not (venv / 'bin/python').exists():
        run([sys.executable, '-m', 'venv', venv])
    lock = ROOT / 'local/python-requirements.txt'
    lines = (ROOT / 'docs/thor/python-lock.txt').read_text().splitlines()
    lock.write_text('\n'.join(line for line in lines if not line.startswith(('-e ', '#'))) + '\n')
    run([venv / 'bin/python', '-m', 'pip', 'install', '-r', lock,
         '--extra-index-url', 'https://download.pytorch.org/whl/cu132'])
    run([venv / 'bin/python', '-m', 'pip', 'install', '--no-deps', '-e', ROOT])
    run([sys.executable, ROOT / 'tools/thor/prepare_tk.py'])
    print('Installed locally. Source tools/thor/env.sh before building/testing.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true', help='download/extract locked packages and populate venv')
    args = parser.parse_args()
    if args.install:
        install()
    else:
        print('Read-only setup check (use --install to populate local/):')
        for relative in ('local/toolchain/usr/local/cuda-13.2/bin/nvcc', 'local/bin/g++',
                         'local/bin/python3-config', 'local/thor-venv/bin/python',
                         'local/thunderkittens/include/common/base_types.cuh'):
            print(f'{relative}: {"PRESENT" if (ROOT / relative).exists() else "MISSING"}')


if __name__ == '__main__':
    main()
