import importlib.util
from types import SimpleNamespace
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('safetensors'),
                     'project tensor dependencies unavailable')
class LatencyScheduleTests(unittest.TestCase):
    def test_upgate_splitting_covers_each_output_once(self):
        from megakernels.demos.latency.scheduler import schedule_upgate
        from megakernels.scheduler import serialize_and_pad
        for workers in (12, 16, 20):
            globs = SimpleNamespace(sm_count=lambda: workers, intermediate_size=8192,
                                    up_gate_proj_block_size=16)
            instructions = schedule_upgate(globs, 0)
            blocks = [block for ins in instructions for block in ins.block_idxs]
            self.assertEqual(sorted(blocks), list(range(512)))
            self.assertTrue(all(len(serialize_and_pad(ins)) == 32 for ins in instructions))

    def test_grouped_oproj_covers_outputs_and_full_wave_fanins(self):
        from megakernels.demos.latency.scheduler import make_dag_layer
        from megakernels.scheduler import Schedule, assign_to_sms, serialize_and_pad
        from megakernels.demos.latency.ready_queue import audit_llama_fanins,pack_dag
        for workers in (12,16,20,40):
            g=SimpleNamespace(sm_count=lambda:workers,skip_attn_reduction=True,ordered_downproj=False,
                o_proj_grouped=True,pos_id=1024,num_attention_heads=32,num_kv_heads=8,head_dim=64,qkv_block_size=16,
                hidden_size=2048,intermediate_size=8192,up_gate_proj_block_size=16,down_proj_block_size=16,
                o_proj_block_size=16,num_hidden_layers=1,vocab_size=128256,lm_head_block_size=32)
            nodes,up=make_dag_layer(g,0,[],'upgate')
            o=[n for n in nodes if n.instruction.opcode()==4]
            self.assertEqual(len(o),workers)
            ranges=[range(n.instruction.start_block_idx,n.instruction.end_block_idx)for n in o]
            self.assertEqual([i for r in ranges for i in r],list(range(128)))
            self.assertLessEqual(max(map(len,ranges))-min(map(len,ranges)),1)
            self.assertTrue(all(set(o)<=set(n.dependencies)for n in up))
            self.assertTrue(all(len(serialize_and_pad(n.instruction))==32 for n in o))
            schedule=Schedule(g,nodes,nodes[-1])
            for mode in ('rr','zz','wave','dag'):assign_to_sms(mode,schedule=schedule)
        from megakernels.demos.latency.scheduler import make_dag
        g.sm_count=lambda:20;g.num_hidden_layers=16
        nodes,end=make_dag(g);schedule=Schedule(g,nodes,end)
        audit_llama_fanins(schedule);packed=pack_dag(nodes)
        self.assertEqual(len(nodes),1428)
        self.assertEqual(len(packed['children']),22400)

    def test_attention_dependencies_cover_q_k_and_v(self):
        from megakernels.demos.latency.scheduler import make_dag_layer
        from megakernels.demos.latency.instructions import PartialAttention
        globs = SimpleNamespace(sm_count=lambda: 20, skip_attn_reduction=True,
                                num_attention_heads=32, num_kv_heads=8, head_dim=64,
                                qkv_block_size=16)
        nodes, outputs = make_dag_layer(globs, 0, [], 'partial')
        self.assertEqual(len(outputs), 8)
        for node in outputs:
            self.assertIsInstance(node.instruction, PartialAttention)
            kv = node.instruction.kv_head_idx
            produced = {block for parent in node.dependencies for block in parent.instruction.block_indices()}
            needed = set(range(kv * 16, (kv + 1) * 16))
            needed.update(range(128 + kv * 4, 132 + kv * 4))
            needed.update(range(160 + kv * 4, 164 + kv * 4))
            self.assertTrue(needed <= produced)

    def test_ordered_down_columns_have_all_producers_and_exact_counter_coverage(self):
        from megakernels.demos.latency.scheduler import make_dag_layer
        for workers in (12, 16, 20):
            globs = SimpleNamespace(sm_count=lambda: workers, skip_attn_reduction=True, ordered_downproj=True,
                                    num_attention_heads=32, num_kv_heads=8, head_dim=64,
                                    qkv_block_size=16, hidden_size=2048, intermediate_size=8192,
                                    up_gate_proj_block_size=16, down_proj_block_size=16, o_proj_block_size=16)
            _, down = make_dag_layer(globs, 0, [], 'downproj')
            for column in range(4):
                current = [n for n in down if n.instruction.reduction_block_idx == column]
                rows = [row for node in current for row in range(node.instruction.start_block_idx,
                                                                 node.instruction.end_block_idx)]
                self.assertEqual(sorted(rows), list(range(128)))
                previous = {n for n in down if n.instruction.reduction_block_idx == column-1}
                for node in current:
                    self.assertTrue(previous <= set(node.dependencies))
