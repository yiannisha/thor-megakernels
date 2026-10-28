import unittest
from unittest.mock import patch
from megakernels.instructions import BaseGlobals

class WorkerBudgetTests(unittest.TestCase):
    @patch.object(BaseGlobals, 'physical_sm_count', return_value=20)
    def test_residency_and_queue_invalidation(self, physical):
        g = BaseGlobals.__new__(BaseGlobals)
        g.__post_init__()
        g.set_active_workers(20)
        instructions, timings = object(), object()
        g.instructions, g.timings = instructions, timings
        for value in (None, True, 1.5, 0, 21):
            with self.assertRaises(ValueError): g.set_active_workers(value)
            self.assertEqual(g.sm_count(), 20)
            self.assertIs(g.instructions, instructions)
            self.assertIs(g.timings, timings)
        g.set_active_workers(3)
        self.assertEqual(g.sm_count(), 3)
        self.assertIsNone(g.instructions)
        self.assertIsNone(g.timings)
