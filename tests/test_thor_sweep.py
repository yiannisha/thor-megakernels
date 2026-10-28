import copy
import unittest
from tools.thor.bench import summarize
from tools.thor.sweep import rank_finalists


def entry(key, values):
    return {'cache_key': key, 'status': 'FINALIST_PASS', 'finalist': {
        'status': 'PASS', 'correctness': {'passed': True},
        'metadata': {'context': 1024, 'input_hash': 'same',
                     'concurrent_workload': 'uncontrolled-existing'},
        'protocol': {'batches': 1, 'repeats': len(values)},
        'trials': [{'samples_ms': values,
                    'correctness_checks': [{'passed': True} for _ in values]}],
        'summary': summarize(values)}}


class SweepTests(unittest.TestCase):
    def test_observed_frontier_never_promotes_shared_gpu_result(self):
        result = rank_finalists([entry('fast', [1., 2.]), entry('slow', [3., 4.])])
        self.assertIsNone(result['selected'])
        self.assertEqual(result['exploratory_fastest'], 'fast')
        self.assertEqual(result['observed_pareto_frontier'], ['fast'])
        self.assertTrue(any('contention' in reason for reason in result['selection_notes']))

    def test_incomparable_and_tampered_results_are_not_ranked(self):
        a, b = entry('a', [1., 2.]), entry('b', [3., 4.])
        b['finalist']['metadata']['input_hash'] = 'different'
        self.assertIsNone(rank_finalists([a, b])['exploratory_fastest'])
        bad = copy.deepcopy(a)
        bad['finalist']['summary']['p50_ms'] = 0.01
        self.assertIsNone(rank_finalists([bad])['exploratory_fastest'])
