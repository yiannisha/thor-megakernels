"""Owned tegrastats collection with conservative, explicitly scoped rail energy."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import threading
import time

from tools.thor.sweep import timed_command


def parse_line(line):
    rails = {name: float(value) * (.001 if unit == 'mW' else 1.)
             for name, value, unit in re.findall(r'\b([A-Z][A-Z0-9_]*)\s+(\d+(?:\.\d+)?)(mW|W)(?=/|\s|$)', line)}
    temperatures = {name: float(value) for name, value in
                    re.findall(r'\b([a-z0-9_]+)@(-?\d+(?:\.\d+)?)C', line)
                    if -100 < float(value) < 200}
    gpu = re.search(r'GR3D_FREQ\s+(\d+)%(?:@(?:\[([^]]+)\]|(\d+)))?', line)
    clocks = None
    if gpu and (gpu[2] or gpu[3]):
        values = (gpu[2] or gpu[3]).split(',')
        clocks = [int(v) for v in values] if all(v.strip().isdigit() for v in values) else None
    return {'rails_w': rails, 'temperatures_c': temperatures,
            'gpu_util_percent': int(gpu[1]) if gpu else None, 'gpu_clocks_mhz': clocks}


def integrate(rows, start, end, interval_s):
    if not all(math.isfinite(x) for x in (start, end, interval_s)) or end <= start or interval_s <= 0:
        raise ValueError('invalid telemetry integration window')
    if any(not math.isfinite(row['monotonic_s']) for row in rows):
        raise ValueError('telemetry timestamps must be finite')
    if any(b['monotonic_s'] <= a['monotonic_s'] for a, b in zip(rows, rows[1:])):
        raise ValueError('telemetry timestamps must strictly increase')
    duration = end - start
    result = {}
    rails = sorted({rail for row in rows for rail in row['rails_w']})
    for rail in rails:
        covered, energy = 0., 0.
        count = sum(start <= row['monotonic_s'] <= end and rail in row['rails_w'] for row in rows)
        for a, b in zip(rows, rows[1:]):
            t0, t1 = a['monotonic_s'], b['monotonic_s']
            if rail not in a['rails_w'] or rail not in b['rails_w'] or t1 - t0 > 3 * interval_s:
                continue
            lo, hi = max(start, t0), min(end, t1)
            if hi <= lo:
                continue
            p0, p1 = a['rails_w'][rail], b['rails_w'][rail]
            if not all(math.isfinite(p) and p >= 0 for p in (p0, p1)):
                continue
            left = p0 + (p1 - p0) * ((lo - t0) / (t1 - t0))
            right = p0 + (p1 - p0) * ((hi - t0) / (t1 - t0))
            covered += hi - lo
            energy += (left + right) * .5 * (hi - lo)
        reasons = []
        if not math.isclose(covered, duration, rel_tol=1e-6, abs_tol=1e-6):
            reasons.append('window is not fully bracketed or contains missing/late samples')
        if count < 10 or duration < 10 * interval_s:
            reasons.append('window is too short for this telemetry sampling interval')
        valid = not reasons
        result[rail] = {'energy_j': energy if valid else None,
                        'mean_power_w': energy / duration if valid else None,
                        'covered_energy_j': energy, 'covered_s': covered, 'window_s': duration,
                        'samples_inside': count, 'valid': valid, 'unavailable_reasons': reasons}
    return result


class Telemetry:
    def __init__(self, path, interval_ms=100):
        if interval_ms < 50 or interval_ms > 10000:
            raise ValueError('interval must be in 50..10000 ms')
        self.path, self.interval_ms = Path(path), interval_ms
        self.source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.rows = []
        self.condition = threading.Condition()
        self.command = ['tegrastats', '--interval', str(interval_ms)]
        if shutil.which('stdbuf'):
            self.command = ['stdbuf', '-oL', *self.command]
        else:
            raise RuntimeError('stdbuf required to bound userspace stdout buffering')

    def start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(self.command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, start_new_session=True, bufsize=1)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        with self.condition:
            ready = self.condition.wait_for(lambda: sum(bool(row['rails_w']) for row in self.rows) >= 2, timeout=5)
        if not ready:
            self.stop()
            raise RuntimeError('tegrastats did not produce two power samples')
        return self

    def _read(self):
        with self.path.open('w') as stream:
            for line in self.process.stdout:
                row = {'monotonic_s': time.monotonic(), 'raw': line.rstrip(), **parse_line(line)}
                with self.condition:
                    self.rows.append(row)
                    stream.write(json.dumps(row) + '\n')
                    stream.flush()
                    self.condition.notify_all()

    def bracket_end(self, end):
        with self.condition:
            return self.condition.wait_for(lambda: bool(self.rows) and self.rows[-1]['monotonic_s'] > end,
                                            timeout=max(2., 3 * self.interval_ms / 1000))

    def stop(self):
        # Never use tegrastats --stop, which could stop another user's collector.
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=2)
        self.thread.join(timeout=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interval-ms', type=int, default=100)
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.timeout < 1:
        parser.error('a bounded command is required after --')
    if args.output.exists():
        raise FileExistsError('choose a fresh output path')
    collector = Telemetry(args.output.with_suffix('.jsonl'), args.interval_ms).start()
    try:
        start = time.monotonic()
        outcome = timed_command(command, args.timeout)
        end = time.monotonic()
        collector.bracket_end(end)
    finally:
        collector.stop()
    result = {'scope': 'whole command, including startup/loading/prefill/checks; not decode-only or per-token energy',
              'command': command, 'process': outcome, 'collector_command': collector.command,
              'start_monotonic_s': start, 'end_monotonic_s': end, 'interval_ms': args.interval_ms,
              'raw_samples': str(collector.path), 'sample_count': len(collector.rows),
              'collector_source_sha256': collector.source_sha256,
              'tegrastats_binary_sha256': hashlib.sha256(Path(shutil.which('tegrastats')).read_bytes()).hexdigest(),
              'rail_estimates': integrate(collector.rows, start, end, args.interval_ms / 1000),
              'timestamp_basis': 'host monotonic receipt; sensor update phase and transport delay are not calibrated',
              'attribution': 'Total observed rail consumption, including other workloads; no idle subtraction',
              'rail_scope': {'VIN': 'system including module and carrier', 'VDD_GPU': 'GPU rail'},
              'gpu_energy_per_token_j': None, 'board_energy_per_token_j': None}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'samples': len(collector.rows),
                      'returncode': outcome['returncode']}))
    return 0 if outcome['returncode'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
