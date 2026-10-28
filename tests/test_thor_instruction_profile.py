import unittest
from tools.experiments.instruction_profile import cycle_delta, extract_rows


class ProfileTests(unittest.TestCase):
    def test_wrap_and_invalid_endpoints(self):
        self.assertEqual(cycle_delta(-10, 10), 20)
        self.assertEqual(cycle_delta(2147483640, -2147483640), 16)
        for a,b in [(0,10), (1,0), (30,20), (1, -2147483647)]:
            with self.assertRaises(ValueError): cycle_delta(a,b)

    def test_role_ownership_and_padding(self):
        stamps = [0]*128
        for event in (5,7,9,11,13,15,17):
            stamps[event:event+2] = [1000+event, 1100+event]
        stamps[44:46] = [-100,100000] # generic diagnostic is never a role duration
        rows = extract_rows([[[1,2], [0]]], [[stamps, [0]*128]], 4, 7)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['consumer_max_cycles'],100)
        self.assertEqual(rows[0]['sample'],7)
        with self.assertRaises(ValueError): extract_rows([[[1]]], [[]],4,0)
        stamps[12]=0
        with self.assertRaises(ValueError): extract_rows([[[1]]], [[stamps]],4,0)
