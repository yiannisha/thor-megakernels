import unittest
from tools.experiments.telemetry import parse_line, integrate


class TelemetryTests(unittest.TestCase):
    def test_rail_units_first_reading_and_missing_gpu_fields(self):
        p = parse_line('VIN 12000mW/10000mW/14000mW VDD_GPU 3W/2W/4W gpu@35C/40C bad@-256C')
        self.assertEqual(p['rails_w'], {'VIN': 12., 'VDD_GPU': 3.})
        self.assertEqual(p['temperatures_c'], {'gpu': 35.})
        self.assertIsNone(p['gpu_clocks_mhz'])
        self.assertEqual(parse_line('GR3D_FREQ 99%@[1098,1098,1098]')['gpu_clocks_mhz'], [1098]*3)

    def test_clipped_trapezoid_and_no_extrapolation(self):
        rows = [{'monotonic_s': i / 10, 'rails_w': {'VIN': 2 + i / 10}} for i in range(31)]
        r = integrate(rows, .05, 2.95, .1)['VIN']
        self.assertTrue(r['valid'])
        self.assertAlmostEqual(r['energy_j'], 10.15)
        self.assertIsNone(integrate(rows, -.1, 3.1, .1)['VIN']['energy_j'])
        self.assertIsNone(integrate(rows, .5, .6, .1)['VIN']['energy_j'])

    def test_missing_rail_late_sample_and_invalid_clock_are_rejected(self):
        rows = [{'monotonic_s': i / 10, 'rails_w': {'VIN': 10.}} for i in range(31)]
        rows[15]['rails_w'] = {}
        self.assertFalse(integrate(rows, 0., 3., .1)['VIN']['valid'])
        gap = rows[:10] + rows[20:]
        self.assertFalse(integrate(gap, 0., 3., .1)['VIN']['valid'])
        with self.assertRaises(ValueError):
            integrate([rows[1], rows[0]], 0., 1., .1)
        with self.assertRaises(ValueError):
            integrate(rows, 1., 0., .1)
