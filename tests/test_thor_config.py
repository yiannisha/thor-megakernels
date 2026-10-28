import unittest
from dataclasses import replace
from tools.thor.config import Config
from tools.thor.sweep import cache_key, candidates, timed_command

DEVICE = {'sm_count': 20, 'cc_major': 11, 'cc_minor': 0}


class ConfigTests(unittest.TestCase):
    def test_first_sweep_and_layout_gate(self):
        configs = list(candidates(Config(), 20))
        self.assertEqual({c.consumer_warps for c in configs}, {4, 8, 16})
        self.assertEqual({c.backoff_ns for c in configs}, {0, 20, 100, 500})
        self.assertEqual({c.workers for c in configs}, {12, 16, 20})
        invalid = {c.name for c in configs if c.validate(DEVICE)}
        self.assertIn('consumer_registers-128', invalid)
        self.assertNotIn('service_registers-32', invalid)
        self.assertTrue(all(c.register_mode == 'dynamic' for c in configs
                            if c.name.startswith(('consumer_registers-', 'service_registers-'))))
        self.assertNotIn('static_registers-128', invalid)
        self.assertFalse(Config().validate(DEVICE))
        for stages in (1, 2, 3):
            self.assertFalse(replace(Config(), weight_stages=stages, page_count=1 + 4 * stages).validate(DEVICE))
            self.assertTrue(replace(Config(), weight_stages=stages, page_count=4 * stages).validate(DEVICE))
        self.assertTrue(replace(Config(), weight_stages=4).validate(DEVICE))

    def test_8k_layout_constraints_and_cache(self):
        for stages in (1, 2, 3):
            config = replace(Config(), page_size=8192, tile_width=256,
                             page_count=1 + 8 * stages, weight_stages=stages)
            self.assertFalse(config.validate(DEVICE))
            self.assertTrue(replace(config, page_count=8 * stages).validate(DEVICE))
            self.assertTrue(replace(config, consumer_warps=4).validate(DEVICE))
            self.assertFalse(replace(config, consumer_warps=8).validate(DEVICE))
            self.assertIn('-DMK_PAGE_SIZE=8192', config.defines())
            self.assertIn('-DMK_WEIGHT_CHUNK_WIDTH=256', config.defines())
            self.assertNotEqual(cache_key(config, {}, 's'), cache_key(Config(), {}, 's'))
        self.assertTrue(replace(Config(), page_size=8192).validate(DEVICE))
        self.assertTrue(replace(Config(), tile_width=256).validate(DEVICE))

    def test_multi_cta_is_explicit_and_residency_is_bounded(self):
        self.assertTrue(replace(Config(), workers=40).validate(DEVICE))
        multi = replace(Config(), allow_multi_cta=True, workers=40)
        self.assertFalse(multi.validate(DEVICE))
        self.assertTrue(replace(multi, workers=41).validate(DEVICE))
        self.assertTrue(replace(multi, register_mode='dynamic').validate(DEVICE))
        self.assertTrue(replace(multi, allow_multi_cta=1).validate(DEVICE))
        self.assertIn('-DMK_ALLOW_MULTI_CTA=1', multi.defines())
        self.assertNotEqual(cache_key(Config(), {}, 's'), cache_key(multi, {}, 's'))
        joined = replace(multi, serial_instruction_handoff=True)
        self.assertFalse(joined.validate(DEVICE))
        self.assertTrue(replace(joined, serial_instruction_handoff=1).validate(DEVICE))
        self.assertIn('-DMK_SERIAL_INSTRUCTION_HANDOFF=1', joined.defines())
        self.assertNotEqual(cache_key(multi, {}, 's'), cache_key(joined, {}, 's'))

    def test_reject_overcommitted_registers_and_workers(self):
        self.assertTrue(replace(Config(), consumer_registers=128).validate(DEVICE))
        self.assertTrue(replace(Config(), workers=21).validate(DEVICE))
        self.assertTrue(replace(Config(), workers=True).validate(DEVICE))

    def test_fp32_downprojection_has_distinct_cache_and_rejects_bf16_ordering(self):
        variant = replace(Config(), fp32_downproj=True)
        self.assertFalse(variant.validate(DEVICE))
        self.assertTrue(replace(variant, ordered_downproj=True).validate(DEVICE))
        self.assertIn("-DMK_FP32_DOWNPROJ=1", variant.defines())
        self.assertNotEqual(cache_key(Config(), {}, "s"), cache_key(variant, {}, "s"))

    def test_compensated_attention_is_explicit_and_cached(self):
        variant = replace(Config(), compensated_attention=True)
        self.assertFalse(variant.validate(DEVICE))
        self.assertTrue(replace(Config(), compensated_attention=1).validate(DEVICE))
        self.assertIn('-DMK_COMPENSATED_ATTENTION=1', variant.defines())
        self.assertNotEqual(cache_key(Config(), {}, 's'), cache_key(variant, {}, 's'))

    def test_tensorcore_matvec_preserves_input_precision_and_cache_identity(self):
        variant = replace(Config(), tensorcore_matvec=True)
        self.assertFalse(variant.validate(DEVICE))
        self.assertTrue(replace(variant, match_torch_bf16_rms=False).validate(DEVICE))
        self.assertTrue(replace(variant, tensorcore_matvec=1).validate(DEVICE))
        self.assertIn('-DMK_TENSORCORE_MATVEC=1', variant.defines())
        self.assertNotEqual(cache_key(Config(), {}, 's'), cache_key(variant, {}, 's'))

    def test_cache_tracks_toolchain_and_sources(self):
        key = cache_key(Config(), {'nvcc': 'a'}, 'source')
        self.assertNotEqual(key, cache_key(Config(), {'nvcc': 'b'}, 'source'))
        self.assertNotEqual(key, cache_key(Config(), {'nvcc': 'a'}, 'modified'))
        self.assertNotEqual(key, cache_key(replace(Config(), backoff_ns=0), {'nvcc': 'a'}, 'source'))

    def test_timeout_terminates_subprocess(self):
        import sys
        result = timed_command([sys.executable, '-c', 'import time; time.sleep(5)'], .05)
        self.assertTrue(result['timeout'])
        self.assertIsNone(result['returncode'])


if __name__ == '__main__':
    unittest.main()
