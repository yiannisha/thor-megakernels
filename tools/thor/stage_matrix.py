"""Build and validate the 1/2/3-stage weight pipelines before comparing latency.

Output staging remains three (up/gate holds a pair). Default layouts use the
minimum 1+4*stages pages; --pages 13 isolates staging from SMEM footprint.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from .config import Config
from .inspect_device import ROOT, inventory
from .sweep import source_hash, timed_command


def write(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stages', nargs='+', type=int, choices=(1, 2, 3), default=[1, 2, 3])
    parser.add_argument('--pages', type=int, help='fixed page budget for staging-only ablation')
    parser.add_argument('--config', type=Path, help='base numerical/warp configuration; stages/pages are overridden')
    parser.add_argument('--sanitizer', choices=('memcheck', 'racecheck', 'synccheck'))
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--context', type=int, default=1024)
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--output', type=Path, required=True, help='directory containing all raw evidence')
    args = parser.parse_args()
    if args.timeout < 1 or args.context < 1:
        parser.error('invalid context/timeout')
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'matrix.json').exists():
        raise FileExistsError('choose a fresh output directory to preserve prior evidence')
    base = Config(**json.loads(args.config.read_text())) if args.config else Config()
    inv = inventory()
    devices = inv['cuda_driver']['devices']
    if not devices:
        parser.error('Thor device required')
    sources = source_hash()
    result = {'timestamp': datetime.now(timezone.utc).isoformat(), 'source_hash': sources,
              'inventory': inv, 'base_config': base.to_dict(), 'variants': [], 'status': 'RUNNING',
              'scope': 'opcode qualification and optional same-cache one-token replay; not autoregressive parity'}
    manifest = directory / 'matrix.json'
    write(manifest, result)
    for stages in dict.fromkeys(args.stages):
        pages = args.pages if args.pages is not None else 1 + 4 * stages
        config = replace(base, name=f'{base.name}-stage-{stages}-pages-{pages}', weight_stages=stages, page_count=pages)
        folder = directory / config.name
        folder.mkdir(exist_ok=True)
        entry = {'config': config.to_dict(), 'status': 'RUNNING', 'errors': config.validate(devices[0]), 'checks': []}
        result['variants'].append(entry)
        write(manifest, result)
        if entry['errors']:
            entry['status'] = 'INVALID'
            continue
        if source_hash() != sources:
            raise RuntimeError('sources changed during matrix; rebuild instead of mixing versions')
        env = os.environ | {'THOR_CONFIG': json.dumps(config.to_dict()), 'THOR_BINARY_DIR': str(folder),
                            'THOR_CONTEXT': str(args.context), 'THOR_ENGINE': 'native'}
        build = ['make', '-C', str(ROOT / 'demos/low-latency-llama'), 'GPU=THOR',
                 f'TARGET={folder / "mk_llama"}', 'EXTRA_NVCCFLAGS=' + ' '.join(config.defines())]
        entry['build'] = timed_command(build, args.timeout, env)
        write(manifest, result)
        if entry['build']['returncode'] != 0:
            entry['status'] = 'BUILD_FAIL'
            continue
        cases = [(op, op, []) for op in ('noop', 'qkv', 'o_proj', 'down', 'upgate', 'lm_head', 'attention', 'partial', 'reduce')]
        cases += [('qkv-k', 'qkv', ['--branch', 'k']), ('qkv-v', 'qkv', ['--branch', 'v']),
                  ('partial-tail', 'partial', ['--context', '81', '--partial-index', '2']),
                  ('reduce-empty-first', 'reduce', ['--empty-first'])]
        cases += [(f'{op}-zero-reuse', op, ['--iterations', '0'])
                  for op in ('qkv', 'o_proj', 'down', 'upgate', 'lm_head')]
        for case, op, options in cases:
            output = folder / f'{case}.json'
            command = [sys.executable, '-m', 'tools.thor.synthetic', '--worker', '--op', op,
                       '--iterations', '8', '--rotary', '--output', str(output), *options]
            if args.sanitizer:
                command = ['compute-sanitizer', '--tool', args.sanitizer, '--error-exitcode', '99', *command]
            process = timed_command(command, args.timeout, env)
            check = {'case': case, 'op': op, 'sanitizer': args.sanitizer, 'process': process,
                     'result': json.loads(output.read_text()) if output.exists() else None}
            entry['checks'].append(check)
            write(manifest, result)
            if process['returncode'] != 0 or not check['result'] or check['result']['status'] != 'PASS':
                entry['status'] = 'CORRECTNESS_FAIL'
                break
        else:
            entry['status'] = 'OPCODES_PASS'
            if args.benchmark:
                output = folder / 'benchmark.json'
                entry['benchmark_process'] = timed_command(
                    [sys.executable, '-m', 'tools.thor.bench', '--adapter', 'tools.thor.model_bench:make_adapter',
                     '--warmups', '5', '--repeats', '30', '--batches', '3', '--output', str(output)],
                    args.timeout, env)
                data = json.loads(output.read_text()) if output.exists() else None
                entry['benchmark'] = data
                entry['status'] = 'PASS' if data and data['status'] == 'PASS' and entry['benchmark_process']['returncode'] == 0 else 'BENCHMARK_FAIL'
        write(manifest, result)
        print(json.dumps({'variant': config.name, 'status': entry['status']}), flush=True)
    good = ('PASS',) if args.benchmark else ('OPCODES_PASS',)
    result['status'] = 'PASS' if all(v['status'] in good for v in result['variants']) else 'FAIL'
    write(manifest, result)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
