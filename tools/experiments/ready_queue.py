"""Build/test a standalone ready-queue prototype; no Llama performance claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from tools.thor.inspect_device import ROOT, command, inventory
from tools.thor.sweep import timed_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stress', action='store_true')
    parser.add_argument('--sanitizer', choices=('memcheck', 'racecheck', 'synccheck'))
    parser.add_argument('--mode', choices=('static', 'ready', 'both'), default='both')
    args = parser.parse_args()
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    report = folder / 'report.json'
    if report.exists():
        raise FileExistsError('choose a fresh output directory to preserve previous evidence')
    source = ROOT / 'demos/ready-queue/probe.cu'
    executable = folder / 'probe'
    build = [os.environ.get('NVCC', 'nvcc'), '-std=c++20', '-O3', '-lineinfo',
             '-gencode', 'arch=compute_110a,code=sm_110a', str(source), '-o', str(executable)]
    result = {'status': 'RUNNING', 'inventory': inventory(), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'build_command': build, 'build': timed_command(build, 180), 'cases': [],
              'concurrent_workload': 'uncontrolled-existing',
              'scope': 'Standalone FP32 vector DAG; no Llama/OpenPI integration',
              'timing_scope': 'kernel CUDA events; resets, CPU reference and output verification excluded',
              'kind': 'correctness-only' if args.stress or args.sanitizer else 'exploratory-kernel-timing',
              'host_compiler': command([os.environ.get('NVCC_CCBIN', 'c++'), '--version'])}
    report.write_text(json.dumps(result, indent=2) + '\n')
    if result['build']['returncode'] != 0:
        result['status'] = 'BUILD_FAIL'
    else:
        result['binary_sha256'] = hashlib.sha256(executable.read_bytes()).hexdigest()
        devices = result['inventory']['cuda_driver']['devices']
        if not devices:
            raise RuntimeError('CUDA device required')
        sms = devices[0]['sm_count']
        if args.stress:
            cases = [(n, w, seed, shape, 500, 1) for n in (1, 17, 257, 4096)
                     for w in (1, 3, sms) for seed in (1, 2) for shape in ('random', 'chain', 'wide')]
        elif args.sanitizer:
            cases = [(257, sms, 1, shape, 500, 1) for shape in ('random', 'chain', 'wide')]
        else:
            cases = [(1024, w, 1, 'random', 500, 30) for w in sorted({min(12, sms), sms})]
        for index, values in enumerate(cases):
            cmd = [str(executable), *map(str, values), args.mode]
            if args.sanitizer:
                cmd = ['compute-sanitizer', '--tool', args.sanitizer, '--error-exitcode', '99', '--log-file',
                       str(folder / f'case-{index}-sanitizer.txt'), *cmd]
            process = timed_command(cmd, 120)
            row = {'command': cmd, 'returncode': process['returncode'], 'timeout': process.get('timeout', False),
                   'output': process['output'], 'status': 'FAIL'}
            if process['returncode'] == 0:
                try:
                    data = json.loads(process['output'])
                    valid = (len(data['variants']) == (2 if args.mode == 'both' else 1)
                             and data['tasks'] == values[0] and data['workers'] == values[1]
                             and all(v['status'] == 'PASS' and v['max_abs'] <= 1e-5
                                     and len(v['samples_ms']) == 3 * values[5] for v in data['variants']))
                    row.update(data=data, status='PASS' if valid else 'FAIL')
                except (ValueError, KeyError, TypeError):
                    pass
            result['cases'].append(row)
            report.write_text(json.dumps(result, indent=2) + '\n')
            if row['status'] != 'PASS':
                break
        result['status'] = 'PASS' if len(result['cases']) == len(cases) and all(c['status'] == 'PASS' for c in result['cases']) else 'FAIL'
    report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'cases': len(result['cases']), 'report': str(report)}))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
