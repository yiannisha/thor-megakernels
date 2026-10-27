"""Execute the exact CUDA arithmetic header using the system C compiler."""
import ctypes
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('gcc')
        if compiler is None:
            raise unittest.SkipTest('C compiler unavailable; actual header tests NOT RUN')
        cls.temp = tempfile.TemporaryDirectory()
        source = Path(cls.temp.name) / 'layout.c'
        source.write_text('#include "page_layout.h"\n'
                          'int release(int n,int s,int p,int i,int q) {return mk_matvec_release_lid(n,s,p,i,q);}\n'
                          'int prefix(int n,int u,int q) {return mk_prefix_release_lid(n,u,q);}\n')
        binary = Path(cls.temp.name) / 'layout.so'
        subprocess.run([compiler, '-std=c11', '-Wall', '-Werror', '-shared', '-fPIC',
                        '-I', str(ROOT / 'include'), str(source), '-o', str(binary)],
                       check=True, capture_output=True)
        cls.lib = ctypes.CDLL(str(binary))
        cls.lib.release.argtypes = [ctypes.c_int] * 5
        cls.lib.prefix.argtypes = [ctypes.c_int] * 3

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_exact_compatibility_permutations(self):
        expected = {1: list(range(5, 13)) + list(range(5)),
                    2: list(range(9, 13)) + list(range(9)),
                    3: list(range(13)),
                    4: [0] + list(range(5, 13)) + list(range(1, 5)),
                    5: [0] + list(range(9, 13)) + list(range(1, 9))}
        for iterations, order in expected.items():
            self.assertEqual([self.lib.release(13, 3, 4, iterations, q) for q in range(13)], order)

    def test_lifetime_order_and_random_reallocation(self):
        rng = random.Random(91)
        for stages in (1, 2, 3):
            for pages in range(1 + 4 * stages, 14):
                physical = list(range(pages))
                for _ in range(300):
                    iterations = rng.randrange(101)
                    last_use = [-1] * pages
                    last_use[0] = 0  # activation consumed before matvec
                    for iteration in range(iterations):
                        for offset in range(4):
                            last_use[1 + (iteration % stages) * 4 + offset] = iteration + 1
                    order = [self.lib.release(pages, stages, 4, iterations, q) for q in range(pages)]
                    self.assertEqual(sorted(order), list(range(pages)))
                    times = [last_use[page] for page in order]
                    self.assertEqual(times, sorted(times))
                    physical = [physical[page] for page in order]
                    self.assertEqual(sorted(physical), list(range(pages)))

    def test_every_opcode_page_release(self):
        # Attention partial uses 2 pages, reduction 1, NoOp returns identity;
        # all other opcodes delegate to the tested matvec release helper.
        for pages in range(5, 14):
            for used in (0, 1, 2):
                actual = [self.lib.prefix(pages, used, q) for q in range(pages)]
                self.assertEqual(actual, list(range(used, pages)) + list(range(used)))

    def test_invalid_layout_and_queries(self):
        for args in [(13, 0, 4, 1, 0), (4, 1, 4, 1, 0), (13, 3, 4, -1, 0),
                     (13, 3, 4, 2, 13), (13, 3, 4, 1, -1)]:
            self.assertEqual(self.lib.release(*args), -1)
