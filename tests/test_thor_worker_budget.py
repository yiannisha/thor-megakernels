import unittest
from unittest.mock import patch
from megakernels.instructions import BaseGlobals


class WorkerBudgetTests(unittest.TestCase):
    @patch.object(BaseGlobals, 'physical_sm_count', return_value=20)
    def test_opt_in_limit_rolls_back_and_invalidates_only_valid_queue_changes(self, physical):
        g = BaseGlobals.__new__(BaseGlobals)
        g.__post_init__()
        g.set_active_workers(20)
        for value in (None, True, 1.5):
            with self.assertRaises(ValueError): g.set_active_workers(value)
        instruction, timing = object(), object()
        g.instructions, g.timings = instruction, timing
        with self.assertRaises(ValueError): g.set_active_workers(21)
        self.assertEqual(g.sm_count(), 20)
        self.assertIs(g.instructions, instruction)
        self.assertIs(g.timings, timing)
        g.set_active_workers(40, max_ctas_per_sm=2)
        self.assertEqual(g.sm_count(), 40)
        self.assertEqual(g.physical_sm_count(), 20)
        self.assertIsNone(g.instructions)
        with self.assertRaises(ValueError): g.set_active_workers(41, max_ctas_per_sm=2)
        self.assertEqual(g.sm_count(), 40)
        self.assertEqual(g.max_ctas_per_sm, 2)
        for value in (0, 3, True):
            with self.assertRaises(ValueError): g.set_active_workers(20, max_ctas_per_sm=value)
        g.set_active_workers(20)
        self.assertEqual(g.max_ctas_per_sm, 1)
