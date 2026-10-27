import importlib.util
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('torch'), 'torch unavailable')
class ReferenceContractTests(unittest.TestCase):
    def test_math_oracle_rejects_lower_precision_settings(self):
        from tools.thor.model_check import attention_context
        with patch('torch.get_float32_matmul_precision', return_value='high'):
            with self.assertRaises(RuntimeError):
                attention_context('math')
        with patch('torch.get_float32_matmul_precision', return_value='highest'), patch(
                'torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed', return_value=True):
            with self.assertRaises(RuntimeError):
                attention_context('math')
        with self.assertRaises(ValueError):
            attention_context('automatic')

    def test_backend_context_restores_optimized_baseline_settings(self):
        import torch
        from tools.thor.model_check import attention_context
        before = (torch.backends.cuda.math_sdp_enabled(), torch.backends.cuda.flash_sdp_enabled())
        with patch('torch.get_float32_matmul_precision', return_value='highest'), patch(
                'torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed', return_value=False):
            with attention_context('math'):
                self.assertTrue(torch.backends.cuda.math_sdp_enabled())
                self.assertFalse(torch.backends.cuda.flash_sdp_enabled())
        self.assertEqual(before, (torch.backends.cuda.math_sdp_enabled(), torch.backends.cuda.flash_sdp_enabled()))
