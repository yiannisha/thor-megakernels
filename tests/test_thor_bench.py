import json
import tempfile
import unittest
from pathlib import Path
from tools.thor.bench import summarize
from tools.thor.report import render


class BenchmarkTests(unittest.TestCase):
    def test_tail_statistics_preserve_outlier(self):
        stats = summarize([1] * 98 + [100, 1000])
        self.assertEqual(stats['p50_ms'], 1)
        self.assertGreater(stats['p99_ms'], stats['p95_ms'])
        self.assertEqual(stats['max_ms'], 1000)

    def test_invalid_measurements_rejected(self):
        for values in ([], [-1], [float('nan')], [float('inf')]):
            with self.assertRaises(ValueError):
                summarize(values)

    def test_failed_runs_never_appear_as_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'failed.json'
            path.write_text(json.dumps({'run_id': 'failure', 'status': 'FAIL',
                                        'summary': {'p50_ms': 1, 'p95_ms': 2, 'p99_ms': 3}}))
            report = render([path])
            self.assertIn('UNAVAILABLE', report)
            self.assertNotIn('1.000000', report)


if __name__ == '__main__':
    unittest.main()
