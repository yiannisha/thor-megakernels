import unittest
from types import SimpleNamespace
from megakernels.scheduler import Schedule,DAG_Node
from megakernels.instructions import NoOp
from tools.experiments.segmented_decode import split_layers


class Instruction:
    def __init__(self,layer):self.layer_idx=layer
    def opcode(self):return 1


class SegmentTests(unittest.TestCase):
    def fixture(self):
        nodes=[]
        for layer in range(5):nodes.append(DAG_Node(Instruction(layer),nodes[-1:]))
        return Schedule(SimpleNamespace(num_hidden_layers=5),nodes,DAG_Node(NoOp(),nodes[-1:]))
    def test_partition_preserves_instructions_and_internal_dependencies(self):
        original=self.fixture();segments=split_layers(original,2)
        self.assertEqual([len(s.dag_nodes) for s in segments],[2,2,1])
        self.assertEqual([id(n.instruction) for s in segments for n in s.dag_nodes],
                         [id(n.instruction) for n in original.dag_nodes])
        for segment in segments:
            members=set(segment.dag_nodes)
            self.assertTrue(all(p in members for n in members for p in n.dependencies))
        self.assertEqual(len(original.dag_nodes[2].dependencies),1)
    def test_future_or_missing_dependency_and_invalid_size_rejected(self):
        source=self.fixture();source.dag_nodes[0].dependencies=[source.dag_nodes[-1]]
        with self.assertRaises(ValueError):split_layers(source,2)
        source=self.fixture();source.dag_nodes[0].dependencies=[DAG_Node(Instruction(0),[])]
        with self.assertRaises(ValueError):split_layers(source,2)
        for size in (0,6,True):
            with self.assertRaises(ValueError):split_layers(self.fixture(),size)
