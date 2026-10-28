"""Record an actual completed native build, including binary hash and exact flags."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex

try:
    from .inspect_device import command
    from .repro import source_hash
except ImportError:
    from inspect_device import command
    from repro import source_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--compiler', required=True)
    parser.add_argument('--gpu', required=True)
    args = parser.parse_args()
    binary = args.binary.resolve()
    with binary.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    flags = os.environ['NVCCFLAGS']
    data = {'timestamp': datetime.now(timezone.utc).isoformat(), 'binary': str(binary),
            'binary_sha256': digest, 'source_hash': source_hash(), 'gpu_target': args.gpu,
            'compiler': command([args.compiler, '--version']), 'flags': shlex.split(flags)}
    Path(str(binary) + '.build.json').write_text(json.dumps(data, indent=2) + '\n')


if __name__ == '__main__':
    main()
