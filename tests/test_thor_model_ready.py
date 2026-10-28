import random
from types import SimpleNamespace
import unittest
import importlib.util


class NoOp:
    def serialize(self): return [0]

from megakernels.scheduler import DAG_Node, Schedule
from megakernels.demos.latency.ready_queue import pack_dag,audit_llama_fanins


class ReadyGraphTests(unittest.TestCase):
    def test_random_ready_only_execution_covers_all_edges(self):
        rng=random.Random(71)
        for count in (0,1,3,33,100):
            nodes=[]
            for _ in range(count):nodes.append(DAG_Node(NoOp(),[n for n in nodes if rng.random()<.15]))
            rng.shuffle(nodes)
            p=pack_dag(nodes);pending=p['indegrees'][:];ready=[i for i,k in enumerate(pending)if k==0];done=set()
            while ready:
                task=ready.pop(rng.randrange(len(ready)))
                self.assertTrue(all(nodes.index(parent)in done for parent in nodes[task].dependencies))
                self.assertNotIn(task,done);done.add(task)
                for child in p['children'][p['offsets'][task]:p['offsets'][task+1]]:
                    pending[child]-=1
                    if pending[child]==0:ready.append(child)
            self.assertEqual(len(done),count)
            self.assertEqual(p['workspace_elements'],3+3*count)
        a=DAG_Node(NoOp(),[]);b=DAG_Node(NoOp(),[a])
        for bad in ([a,a],[b]):
            with self.assertRaises(ValueError):pack_dag(bad)
        b.dependencies=[a,a]
        with self.assertRaises(ValueError):pack_dag([a,b])
        b.dependencies=[a];a.dependencies=[b]
        with self.assertRaises(ValueError):pack_dag([a,b])

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('safetensors'), 'tensor dependencies unavailable')
    def test_actual_llama_barrier_coverage_rejects_missing_edges_and_outputs(self):
        from megakernels.demos.latency.scheduler import make_dag
        g=SimpleNamespace(sm_count=lambda:20,skip_attn_reduction=True,ordered_downproj=False,
                          num_attention_heads=32,num_kv_heads=8,head_dim=64,qkv_block_size=16,
                          hidden_size=2048,intermediate_size=8192,up_gate_proj_block_size=16,
                          down_proj_block_size=16,o_proj_block_size=16,num_hidden_layers=2,
                          vocab_size=128256,lm_head_block_size=32)
        nodes,end=make_dag(g);s=Schedule(g,nodes,end)
        audit_llama_fanins(s);pack_dag(nodes)
        for op in (1,2,4,5,6,7):
            node=next(n for n in nodes if n.instruction.opcode()==op and n.dependencies)
            original=node.dependencies;node.dependencies=[]
            with self.assertRaises(ValueError):audit_llama_fanins(s)
            node.dependencies=original
        node=next(n for n in nodes if n.instruction.opcode()==4)
        original=node.instruction.end_block_idx;node.instruction.end_block_idx+=1
        with self.assertRaises(ValueError):audit_llama_fanins(s)
        node.instruction.end_block_idx=original
        from dataclasses import replace
        for op in (1,4,5,6,7):
            template=next(n for n in nodes if n.instruction.opcode()==op)
            ins=template.instruction
            if op==5:empty=replace(ins,block_idxs=[])
            elif op in (1,7):empty=replace(ins,end_output_block_idx=ins.start_output_block_idx)
            else:empty=replace(ins,end_block_idx=ins.start_block_idx)
            extra=DAG_Node(empty,list(template.dependencies));nodes.append(extra)
            with self.assertRaises(ValueError):audit_llama_fanins(s)
            nodes.pop()

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch unavailable')
    def test_non_native_ready_adapter_rejected_before_loading_weights(self):
        import json
        from unittest.mock import patch
        from tools.thor.model_bench import ModelAdapter
        config=json.dumps({'ready_queue':True,'serial_instruction_handoff':True})
        with patch.dict('os.environ',{'THOR_CONFIG':config}), patch('tools.thor.model_bench.load_reference',side_effect=AssertionError('weights must not load')):
            for engine in ('torch','torch_graph','pyvm'):
                with self.assertRaisesRegex(ValueError,'native'):ModelAdapter(engine)
