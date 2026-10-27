import unittest
from types import SimpleNamespace


class GeneratorPositions(unittest.TestCase):
    def test_decode_sequence_length_matches_current_position(self):
        import torch
        from megakernels.generators import PyTorchGenerator
        calls = []
        class Model:
            device = 'cpu'
            def __call__(self, state):
                calls.append((state.position_ids.item(), state.seq_len, state.input_ids.item()))
                return SimpleNamespace(output_ids=state.input_ids + 1)
        tokens = torch.tensor([[10, 11, 0, 0, 0]])
        PyTorchGenerator(Model()).generate(tokens, prompt_len=7, ntok=3, ntok_already_generated=2)
        self.assertEqual(calls, [(8, 9, 11), (9, 10, 12), (10, 11, 13)])
        self.assertEqual(tokens.tolist(), [[10, 11, 12, 13, 14]])
