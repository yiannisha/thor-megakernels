"""Build and run the model-free Thor VM smoke test, with an external watchdog."""
import argparse
import json
import os
from pathlib import Path
import subprocess

try:
    from .inspect_device import ROOT, inventory
    from .sweep import source_hash, timed_command
except ImportError:
    from inspect_device import ROOT, inventory
    from sweep import source_hash, timed_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='compile and launch; default only prints build command')
    parser.add_argument('--warps', type=int, choices=(4, 8, 16), default=16)
    parser.add_argument('--sanitizer', choices=('memcheck', 'racecheck', 'synccheck', 'initcheck'))
    parser.add_argument('--timeout', type=int, default=60)
    parser.add_argument('--output', type=Path, default=ROOT / 'local/thor-smoke-result.json')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('timeout must be positive')
    binary = ROOT / f'local/thor-smoke-w{args.warps}'
    build = [os.environ.get('NVCC', 'nvcc'), '-std=c++20', '-O3', '-lineinfo', '-gencode=arch=compute_110a,code=sm_110a', '-maxrregcount=96',
             '-DKITTENS_HOPPER', '-DMK_THOR', f'-DMK_CONSUMER_WARPS={args.warps}',
             '--expt-relaxed-constexpr', '--expt-extended-lambda',
             '-I' + os.environ.get('THUNDERKITTENS_ROOT', str(ROOT / 'ThunderKittens')) + '/include',
             '-I' + str(ROOT / 'include'), str(ROOT / 'tools/thor/smoke.cu'), '-lcuda', '-o', str(binary)]
    result = {'status': 'DRY_RUN', 'build_command': build, 'inventory': inventory(),
              'source_hash': source_hash(), 'consumer_warps': args.warps, 'sanitizer': args.sanitizer, 'timeout_seconds': args.timeout}
    if args.execute:
        binary.parent.mkdir(parents=True, exist_ok=True)
        outcome = timed_command(build, args.timeout)
        result['build'] = outcome
        result['status'] = 'BUILD_FAIL'
        if outcome['returncode'] == 0:
            command = [str(binary)]
            if args.sanitizer:
                command = ['compute-sanitizer', '--tool', args.sanitizer, '--error-exitcode', '99', *command]
            result['run'] = timed_command(command, args.timeout)
            result['status'] = 'PASS' if result['run']['returncode'] == 0 else 'FAIL'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output)}))
    return 0 if result['status'] in ('PASS', 'DRY_RUN') else 1


if __name__ == '__main__':
    raise SystemExit(main())
