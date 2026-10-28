import copy
from pathlib import Path
import unittest
from tools.thor.autoregressive_report import qualification_note


class QualificationBindingTests(unittest.TestCase):
    def setUp(self):
        meta = {'context': 5, 'input_hash': 'input', 'config': {'name': 'test'},
                'prefill_attention': 'flash', 'reference_attention': 'math',
                'model_provenance': {'weights': [{'file': 'weights', 'sha256': 'weights-sha'}]},
                'native_build': {'binary_sha256': 'binary-sha'}, 'decode_steps_after_prefill': 2}
        metrics = {key: {'passed': True, 'normalized_rmse': .001, 'nrmse_limit': .02}
                   for key in ('logits', 'hidden', 'k_cache', 'v_cache')}
        self.native = {'metadata': copy.deepcopy(meta), 'trials': [{'requests': [{'generated_tokens': [10, 11, 12]}]}]}
        self.qual = {'metadata': meta, 'status': 'PASS', 'protocol': {'native_past': True},
                     'rows': [{'input_token': x, 'native_token': x+1, 'passed': True,
                               'metrics': copy.deepcopy(metrics)} for x in (10, 11)]}

    def test_matching_qualification(self):
        self.assertIn('0/2 failed steps', qualification_note(self.qual, self.native, Path('q.json')))

    def test_different_binary_and_trajectory_are_rejected(self):
        for field in ('binary', 'trajectory', 'weights', 'gate'):
            q = copy.deepcopy(self.qual)
            if field == 'binary': q['metadata']['native_build']['binary_sha256'] = 'other'
            if field == 'trajectory': q['rows'][0]['native_token'] = 99
            if field == 'weights': q['metadata']['model_provenance']['weights'][0]['sha256'] = 'other'
            if field == 'gate': q['rows'][0]['metrics']['logits']['normalized_rmse'] = .5
            with self.subTest(field=field), self.assertRaises(ValueError):
                qualification_note(q, self.native, Path('q.json'))
