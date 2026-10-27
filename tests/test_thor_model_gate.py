import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch unavailable')
class ModelGateTests(unittest.TestCase):
    def test_user_logit_gate_accepts_218_percent_but_state_gate_stays_strict(self):
        import torch
        from tools.thor.model_check import errors
        reference = torch.ones(128)
        output = reference * 1.0218
        self.assertTrue(errors(output, reference, nrmse_limit=.03)['passed'])
        self.assertFalse(errors(output, reference)['passed'])

    def test_logit_nrmse_does_not_waive_maximum_error(self):
        import torch
        from tools.thor.model_check import errors
        reference = torch.ones(10000)
        output = reference.clone()
        output[0] = 1.5
        result = errors(output, reference, nrmse_limit=.03)
        self.assertLess(result['normalized_rmse'], .03)
        self.assertFalse(result['passed'])
