import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'torch unavailable')
class IsolatedDependencies(unittest.TestCase):
    def test_lm_head_retains_completed_downprojection_dependency(self):
        import torch
        from tools.thor.synthetic import reset_isolated_barriers
        barriers = torch.zeros((16, 10, 48), dtype=torch.int32)
        reset_isolated_barriers(barriers, 'lm_head', True)
        self.assertEqual(barriers[15, 5, 0].item(), 512)
        self.assertTrue(bool((barriers == 512).all()))

    def test_fp32_downprojection_clears_own_counters_only(self):
        import torch
        from tools.thor.synthetic import reset_isolated_barriers
        barriers = torch.zeros((16, 10, 48), dtype=torch.int32)
        reset_isolated_barriers(barriers, 'down', True)
        self.assertTrue(bool((barriers[:, 5] == 0).all()))
        self.assertTrue(bool((barriers[:, 7:10] == 0).all()))
        self.assertTrue(bool((barriers[:, 4] == 512).all()))
        reset_isolated_barriers(barriers, 'down', False)
        self.assertTrue(bool((barriers == 512).all()))
