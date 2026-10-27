"""Explicit opt-in; CUDA tests never download weights or alter power settings."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


@unittest.skipUnless(os.environ.get('MK_RUN_CUDA_TESTS') == '1', 'set MK_RUN_CUDA_TESTS=1 after sourcing tools/thor/env.sh')
class CudaSmokeTests(unittest.TestCase):
    def test_vm_smoke_memcheck(self):
        root = Path(__file__).resolve().parents[1]
        subprocess.run([sys.executable, str(root / 'tools/thor/smoke.py'), '--execute',
                        '--sanitizer', 'memcheck'], cwd=root, check=True, timeout=150)
