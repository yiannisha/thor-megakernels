"""Actual Llama layer-boundary segmentation, optionally captured in a CUDA Graph.

Graph replay currently qualifies fixed-position, frozen-cache decode only.
"""
import hashlib
import math
import os
from pathlib import Path


def split_layers(schedule, layers_per_segment):
    from megakernels.scheduler import Schedule,DAG_Node
    from megakernels.instructions import NoOp
    total=schedule.globs.num_hidden_layers
    if type(layers_per_segment) is not int or not 1<=layers_per_segment<=total:
        raise ValueError('invalid layer segment size')
    nodes=schedule.dag_nodes
    if len(set(nodes))!=len(nodes):raise ValueError('duplicate source DAG node')
    groups={}
    indices={}
    for node in nodes:
        layer=getattr(node.instruction,'layer_idx',total-1 if node.instruction.opcode()==7 else -1)
        if not 0<=layer<total:raise ValueError('instruction has no valid layer')
        index=layer//layers_per_segment
        groups.setdefault(index,[]).append(node);indices[node]=index
    for node in nodes:
        for parent in node.dependencies:
            if parent not in indices:raise ValueError('missing source dependency')
            if indices[parent]>indices[node]:raise ValueError('dependency crosses a segment backwards')
    segments=[]
    for index in sorted(groups):
        members=groups[index];mapping={n:DAG_Node(n.instruction,[]) for n in members}
        for node in members:mapping[node].dependencies=[mapping[p] for p in node.dependencies if p in mapping]
        parents={p for n in members for p in n.dependencies if p in mapping}
        leaves=[mapping[n] for n in members if n not in parents]
        segments.append(Schedule(schedule.globs,[mapping[n] for n in members],DAG_Node(NoOp(),leaves)))
    if sum(len(s.dag_nodes) for s in segments)!=len(nodes):raise ValueError('segment coverage failure')
    return segments


class SegmentedAdapter:
    def __init__(self):
        import torch
        from tools.thor.model_bench import make_adapter as base_factory
        from megakernels.scheduler import assign_to_sms,tensorize_instructions
        self.base=base_factory()
        self.layers=int(os.environ.get('THOR_SEGMENT_LAYERS','4'))
        self.mode=os.environ.get('THOR_SEGMENT_MODE','eager')
        if self.mode not in ('eager','graph'):raise ValueError('mode must be eager|graph')
        g=self.base.globs
        original_queues=(g.instructions,g.timings)
        self.queues=[]
        self.segments=split_layers(self.base.schedule,self.layers)
        for schedule in self.segments:
            tensorize_instructions(g,assign_to_sms(self.base.config.scheduler,schedule=schedule))
            self.queues.append((g.instructions,g.timings))
        g.instructions,g.timings=original_queues
        original_interpret=self.base.interpret
        def interpret(globs,native):
            before=(globs.instructions,globs.timings)
            try:
                for instructions,timings in self.queues:
                    globs.instructions,globs.timings=instructions,timings
                    original_interpret(globs,native)
            finally:globs.instructions,globs.timings=before
        self.base.interpret=interpret
        self.metadata=dict(self.base.metadata,variant=f'native:segments-{self.layers}-{self.mode}',
                           architecture='layer_segmented_'+self.mode,segment_count=len(self.segments),
                           layers_per_segment=self.layers,segment_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           boundary_contract='Prior segment completes in the same CUDA stream; global activations and barriers persist between kernels',
                           graph_limit='fixed decode position only; advancing-position autoregressive graph execution not implemented')
        if self.mode=='graph':
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):self.base.reset();self.base.run()
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            self.base.reset()
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):self.output=self.base.run()
        self.reset()

    def __getattr__(self,name):return getattr(self.base,name)
    def reset(self):self.base.reset()
    def run(self):
        if self.mode=='graph':self.graph.replay();return self.output
        return self.base.run()


def make_adapter():return SegmentedAdapter()
