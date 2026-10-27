"""Constraint-aware build/benchmark sweep. Dry-run by default; no installations.

Adapters receive THOR_CONFIG (JSON) and THOR_BINARY_DIR. They must apply scheduler
and worker settings, reset inputs, and attest exact configuration in metadata.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

try:
    from .config import Config
    from .inspect_device import ROOT, command, inventory
except ImportError:
    from config import Config
    from inspect_device import ROOT, command, inventory


def timed_command(argv, timeout, env=None):
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, env=env, start_new_session=True, cwd=ROOT)
    try:
        output, _ = process.communicate(timeout=timeout)
        return {'returncode': process.returncode, 'output': output}
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        return {'returncode': None, 'timeout': True, 'output': output}


def source_hash():
    digest = hashlib.sha256()
    for folder in ('include', 'demos/low-latency-llama', 'ThunderKittens/include', 'megakernels', 'tools/thor'):
        for path in sorted((ROOT / folder).rglob('*')):
            if path.is_file() and (path.suffix in ('.cu', '.cuh', '.h', '.py', '.sh') or path.name == 'Makefile'):
                digest.update(str(path.relative_to(ROOT)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def candidates(base, sms):
    # One-dimensional ablations preserve interpretability before joint tuning.
    yield base
    dimensions = {'consumer_warps': [4, 8, 16], 'consumer_registers': [64, 96, 104, 128],
                  'service_registers': [32, 48, 64], 'static_registers': [32, 64, 96, 128], 'workers': [x for x in (12, 16, 20) if x <= sms],
                  'backoff_ns': [0, 20, 100, 500], 'scheduler': ['rr', 'zz', 'wave', 'dag']}
    for dimension, values in dimensions.items():
        for value in values:
            if value != getattr(base, dimension):
                changes = {dimension: value}
                if dimension in ('consumer_registers', 'service_registers'):
                    changes['register_mode'] = 'dynamic'
                yield replace(base, name=f'{dimension}-{value}', **changes)


def cache_key(config, toolchain, sources):
    return hashlib.sha256(json.dumps({'config': config.to_dict(), 'toolchain': toolchain,
                                     'sources': sources}, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/thor/upstream-like.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'local/thor-sweep')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--adapter')
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--retries', type=int, default=1)
    args = parser.parse_args()
    if args.timeout < 1 or args.retries < 0:
        parser.error('timeout must be positive, retries nonnegative')
    if args.execute and not args.adapter:
        parser.error('--execute requires --adapter module:factory')
    inv = inventory()
    devices = inv['cuda_driver']['devices']
    if not devices:
        parser.error('no CUDA driver device; supply hardware before a device-constrained sweep')
    base = Config(**json.loads(args.config.read_text()))
    sources = source_hash()
    toolchain = {'nvcc': command(['nvcc', '--version']), 'python': inv['python'],
                 'packages': inv['packages'], 'targets': command(['nvcc', '--list-gpu-code']),
                 'host_compiler': command(['c++', '--version'])}
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {'inventory': inv, 'toolchain': toolchain, 'source_hash': sources, 'variants': []}
    for config in candidates(base, devices[0]['sm_count']):
        key = cache_key(config, toolchain, sources)
        entry = {'config': config.to_dict(), 'cache_key': key, 'errors': config.validate(devices[0]),
                 'status': 'PLANNED', 'attempts': []}
        manifest['variants'].append(entry)
        if entry['errors']:
            entry['status'] = 'INVALID'
        elif args.execute:
            targets = toolchain['targets'].get('stdout', '').split()
            if config.arch.removesuffix('a') not in targets or not inv['packages']['pybind11']:
                entry['status'] = 'BLOCKED'
                entry['errors'].append('compiler target or pybind11 prerequisite unavailable')
            else:
                build_dir = args.output.resolve() / key
                build_dir.mkdir(exist_ok=True)
                env = os.environ | {'THUNDERKITTENS_ROOT': os.environ.get('THUNDERKITTENS_ROOT', str(ROOT / 'ThunderKittens')),
                                    'MEGAKERNELS_ROOT': str(ROOT), 'THOR_CONFIG': json.dumps(config.to_dict()),
                                    'THOR_BINARY_DIR': str(build_dir)}
                build = ['make', '-B', '-C', str(ROOT / 'demos/low-latency-llama'), 'GPU=THOR',
                         f'THOR_ARCH={config.arch}', f'THOR_WORKERS={config.workers}',
                         f'THOR_STATIC_REGISTERS={config.static_registers}', f'THOR_REGISTER_MODE={config.register_mode}',
                         f'TARGET={build_dir / "mk_llama"}', 'EXTRA_NVCCFLAGS=' + ' '.join(config.defines())]
                cached = build_dir / 'build-success.json'
                if cached.exists() and list(build_dir.glob('mk_llama*.so')):
                    entry['build_cached'] = True
                else:
                    for attempt in range(args.retries + 1):
                        outcome = timed_command(build, args.timeout, env)
                        entry['attempts'].append({'phase': 'build', **outcome})
                        if outcome['returncode'] == 0:
                            cached.write_text(json.dumps({'key': key, 'command': build}))
                            break
                    else:
                        entry['status'] = 'BUILD_FAIL'
                if entry['status'] == 'PLANNED':
                    output = build_dir / 'pilot.json'
                    outcome = timed_command([sys.executable, str(ROOT / 'tools/thor/bench.py'),
                                             '--adapter', args.adapter, '--repeats', '10', '--batches', '1',
                                             '--output', str(output)], args.timeout, env)
                    entry['attempts'].append({'phase': 'pilot', **outcome})
                    entry['status'] = 'PILOT_FAIL'
                    if outcome['returncode'] == 0 and output.exists():
                        data = json.loads(output.read_text())
                        if data.get('metadata', {}).get('config') != config.to_dict():
                            entry['errors'].append('adapter did not attest exact applied configuration')
                        elif data['status'] == 'PASS':
                            entry['status'] = 'PILOT_PASS'
                            entry['pilot'] = data
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    finalists = sorted((e for e in manifest['variants'] if e['status'] == 'PILOT_PASS'),
                       key=lambda e: e['pilot']['summary']['p50_ms'])[:3]
    for entry in finalists:
        build_dir = args.output.resolve() / entry['cache_key']
        env = os.environ | {'THOR_CONFIG': json.dumps(entry['config']), 'THOR_BINARY_DIR': str(build_dir)}
        outcome = timed_command([sys.executable, str(ROOT / 'tools/thor/bench.py'), '--adapter', args.adapter,
                                 '--output', str(build_dir / 'finalist.json')], args.timeout, env)
        entry['attempts'].append({'phase': 'finalist', **outcome})
        entry['status'] = 'FINALIST_FAIL'
        if outcome['returncode'] == 0:
            data = json.loads((build_dir / 'finalist.json').read_text())
            if data.get('metadata', {}).get('config') == entry['config'] and data['status'] == 'PASS':
                entry['status'] = 'FINALIST_PASS'
                entry['finalist'] = data
    manifest['selected'] = None
    valid = [e for e in finalists if e['status'] == 'FINALIST_PASS']
    if valid:
        manifest['selected'] = min(valid, key=lambda e: e['finalist']['summary']['p50_ms'])['cache_key']
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'variants': len(manifest['variants']), 'selected': manifest['selected'],
                      'manifest': str(args.output / 'manifest.json')}))


if __name__ == '__main__':
    main()
