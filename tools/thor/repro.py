"""Source fingerprints and watchdogs for reproducible Thor builds and runs."""
import hashlib
import os
import signal
import subprocess
from pathlib import Path
try:
    from .inspect_device import ROOT
except ImportError:
    from inspect_device import ROOT

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
