"""Static PiZero range coverage for the smaller Thor worker budget."""
import importlib.util
from pathlib import Path
import sys
import unittest

spec = importlib.util.spec_from_file_location(
    'pi0_program', Path(__file__).resolve().parents[1]/'demos/pi0/native_pi0/program.py')
program = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = program
spec.loader.exec_module(program)


class Pi0ProgramTest(unittest.TestCase):
    def test_dependent_stages_have_complete_unique_monotonic_ranges(self):
        stages = [program.LinearWork(tiles, k) for tiles, k in
                  [(1, 64), (120, 2048), (20, 1024), (16, 4096), (768, 16384)]]
        for workers in (1, 8, 20):
            code = program.linear_streams(stages, workers)
            coverage = [[] for _ in stages]
            for stream in code:
                prior = -1
                for opcode, stage, first, count, *_ in stream:
                    if not opcode:
                        continue
                    self.assertGreaterEqual(stage, prior)
                    prior = stage
                    coverage[stage].extend(range(first, first+count))
            for work, actual in zip(stages, coverage):
                self.assertEqual(sorted(actual), list(range(work.tiles)))


if __name__ == '__main__':
    unittest.main()
