"""Read-only inventory; works without PyTorch or a CUDA toolkit."""
import argparse
import ctypes as C
import ctypes.util
import importlib.metadata
import json
import platform
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=15)
        return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": str(e)}


def driver_devices():
    devices = []
    try:
        lib = C.CDLL(C.util.find_library('cuda') or 'libcuda.so.1')
        def check(rc):
            if rc:
                raise RuntimeError(f'CUDA driver error {rc}')
        check(lib.cuInit(0))
        count = C.c_int()
        check(lib.cuDeviceGetCount(C.byref(count)))
        for ordinal in range(count.value):
            dev = C.c_int()
            check(lib.cuDeviceGet(C.byref(dev), ordinal))
            name = C.create_string_buffer(256)
            check(lib.cuDeviceGetName(name, len(name), dev))
            attrs = {}
            for key, attr in {'sm_count': 16, 'cc_major': 75, 'cc_minor': 76,
                              'max_threads_per_block': 1, 'shared_memory_per_block': 8,
                              'shared_memory_per_sm': 81, 'shared_memory_optin': 97,
                              'max_blocks_per_sm': 106, 'cooperative_launch': 95}.items():
                value = C.c_int()
                rc = lib.cuDeviceGetAttribute(C.byref(value), attr, dev)
                attrs[key] = value.value if rc == 0 else None
            devices.append({'ordinal': ordinal, 'name': name.value.decode(), **attrs})
    except (OSError, RuntimeError) as e:
        return {'devices': devices, 'error': str(e)}
    return {'devices': devices}


def inventory():
    versions = {}
    for package in ('torch', 'transformers', 'pybind11'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    commands = {'nvcc': ['nvcc', '--version'], 'nvcc_targets': ['nvcc', '--list-gpu-code'],
                'nvidia_smi': ['nvidia-smi', '-q'], 'power_mode': ['nvpmodel', '-q'],
                'git_sha': ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                'worktree': ['git', '-C', str(ROOT), 'status', '--short'],
                'submodules': ['git', '-C', str(ROOT), 'submodule', 'status']}
    return {'platform': platform.platform(), 'python': platform.python_version(),
            'packages': versions, 'cuda_driver': driver_devices(),
            'tools': {x: shutil.which(x) for x in ('nvcc', 'tegrastats', 'nsys', 'ncu', 'jtop')},
            'commands': {k: command(v) for k, v in commands.items()},
            'jetpack_release': Path('/etc/nv_tegra_release').read_text() if Path('/etc/nv_tegra_release').exists() else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = json.dumps(inventory(), indent=2)
    if args.output:
        args.output.write_text(result + '\n')
    print(result)


if __name__ == '__main__':
    main()
