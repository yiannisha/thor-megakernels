"""Export only unique, same-warp timing pairs from actual Llama decode.

Durations are SM clock cycles, not a global timeline or milliseconds. Consumer
spans include waits and synchronization and must not be labeled compute time.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def cycle_delta(start, end):
    """Decode int32 timestamps modulo 2**32; reject missing/ambiguous pairs."""
    if start == 0 or end == 0:
        raise ValueError('missing timing endpoint')
    delta = (int(end) - int(start)) & 0xffffffff
    if delta >= 0x80000000:
        raise ValueError('ambiguous or reversed timing pair')
    return delta


def extract_rows(instructions, timings, consumer_warps, sample):
    if len(instructions) != len(timings):
        raise ValueError('worker dimensions differ')
    roles = [('loader', 5), ('launcher', 7), ('storer', 9)]
    roles += [(f'consumer_{w}', 11 + 2*w) for w in range(consumer_warps)]
    rows = []
    for worker, (queue, events) in enumerate(zip(instructions, timings)):
        if len(queue) != len(events):
            raise ValueError('instruction dimensions differ')
        for index, (instruction, stamps) in enumerate(zip(queue, events)):
            if len(stamps) != 128:
                raise ValueError('expected timing schema width 128')
            if instruction[0] == 0:
                continue
            spans = {role: cycle_delta(stamps[event], stamps[event+1]) for role, event in roles}
            rows.append({'sample': sample, 'worker': worker, 'queue_index': index,
                         'instruction': instruction, 'opcode': instruction[0],
                         'role_cycles': spans,
                         'consumer_max_cycles': max(spans[f'consumer_{w}'] for w in range(consumer_warps))})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.context, args.samples) < 1 or args.warmups < 0:
        parser.error('positive context/samples and nonnegative warmups required')
    import torch
    from tools.thor.model_bench import ModelAdapter
    with torch.inference_mode():
        adapter = ModelAdapter('native', args.context, args.seed)
        if not adapter.config.timing:
            raise ValueError('requires a build and THOR_CONFIG with timing=true')
        adapter.reset(); output = adapter.run(); torch.cuda.synchronize()
        check = adapter.check(output, adapter.reference())
        if not check['passed']:
            raise RuntimeError('pre-profile numerical check failed')
        for _ in range(args.warmups):
            adapter.reset(); adapter.run()
        torch.cuda.synchronize()
        instructions = adapter.globs.instructions.cpu().tolist()
        from megakernels.scheduler import serialize_and_pad
        default_costs = {tuple(serialize_and_pad(node.instruction)): node.instruction.cost(adapter.globs)
                         for node in adapter.schedule.dag_nodes}
        rows = []
        checks = []
        for sample in range(args.samples):
            adapter.reset(); adapter.globs.timings.zero_()
            output = adapter.run(); torch.cuda.synchronize()
            checks.append(adapter.check(output, adapter.reference()))
            sample_rows = extract_rows(instructions, adapter.globs.timings.cpu().tolist(),
                                       adapter.config.consumer_warps, sample)
            for row in sample_rows:
                row['default_cost'] = default_costs[tuple(row['instruction'])]
            rows.extend(sample_rows)
        groups = {}
        for row in rows:
            groups.setdefault(str(row['opcode']), []).append(row['consumer_max_cycles'])
        result = {'schema_version': 'thor.instruction-profile.v1',
                  'status': 'PASS' if all(c['passed'] for c in checks) else 'FAIL',
                  'metadata': adapter.metadata,
                  'protocol': vars(args) | {'output': str(args.output)},
                  'units': 'SM clock cycles; only endpoints from the same warp are subtracted',
                  'limitations': ['Inclusive role spans contain waits; not isolated execution costs',
                                 'No cross-worker or cross-role timestamp subtraction',
                                 'Instrumentation overhead and shared GPU load can bias costs'],
                  'checks': checks, 'rows': rows,
                  'opcode_consumer_max_medians': {op: statistics.median(v) for op,v in groups.items()},
                  'probe_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps({'status': result['status'], 'rows': len(rows), 'output': str(args.output)}))
        if result['status'] != 'PASS':
            raise RuntimeError('profile numerical qualification failed')


if __name__ == '__main__':
    main()
