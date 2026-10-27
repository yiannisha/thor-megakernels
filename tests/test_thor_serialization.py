"""CPU tensor tests when project dependencies are installed; no GPU launch."""
import importlib.util
from types import SimpleNamespace
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('safetensors'),
                     'optional Python tensor dependencies unavailable')
class SerializationTests(unittest.TestCase):
    def globals(self, workers):
        return SimpleNamespace(sm_count=lambda: workers, device='cpu')

    def test_nonsquare_queues_padding_and_input_immutability(self):
        from megakernels.instructions import NoOp
        from megakernels.scheduler import tensorize_instructions
        queues = [[NoOp()], [NoOp(), NoOp(), NoOp()], []]
        globs = self.globals(3)
        tensorize_instructions(globs, queues)
        self.assertEqual(tuple(globs.instructions.shape), (3, 3, 32))
        self.assertEqual(tuple(globs.timings.shape), (3, 3, 128))
        self.assertEqual([len(q) for q in queues], [1, 3, 0])
        self.assertEqual(globs.instructions.count_nonzero().item(), 0)

    def test_empty_queues_have_explicit_zero_length(self):
        from megakernels.scheduler import tensorize_instructions
        globs = self.globals(2)
        tensorize_instructions(globs, [[], []])
        self.assertEqual(tuple(globs.instructions.shape), (2, 0, 32))

    def test_invalid_queue_count_and_oversize_instruction(self):
        from megakernels.scheduler import tensorize_instructions, serialize_and_pad
        with self.assertRaises(ValueError):
            tensorize_instructions(self.globals(2), [[]])
        with self.assertRaises(ValueError):
            serialize_and_pad(SimpleNamespace(serialize=lambda: [1] * 33))
