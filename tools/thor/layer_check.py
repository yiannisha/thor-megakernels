"""Locate numerical drift using native layer prefixes and PyTorch layer inputs.

Each native prefix starts from the same reference layer input and past KV, so
errors here do not include propagation from preceding native layers.
"""
import argparse
import json
from pathlib import Path


def main():
    import torch
    from .model_bench import ModelAdapter
    from .model_check import errors
    from .teacher_forced import write
    from megakernels.scheduler import Schedule, DAG_Node, assign_to_sms, tensorize_instructions
    from megakernels.instructions import NoOp
    from megakernels.demos.latency.scheduler import make_dag_layer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', type=int, default=1024)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.step < 0 or args.context < 1:
        parser.error('invalid position')
    report = {'scope': 'native prefixes with reference layer inputs and past KV', 'rows': [],
              'protocol': vars(args) | {'output': str(args.output)}}
    with torch.inference_mode():
        a = ModelAdapter('native', args.context, args.seed, cache_capacity=args.context + args.step + 1)
        report['metadata'] = a.metadata
        a.reset()
        a.engine = 'torch'
        for step in range(args.step):
            a.context = args.context + step
            a.position.fill_(a.context)
            a.state.seq_len = a.context + 1
            token = a.run().argmax()
            a.case.token.copy_(token.reshape(1, 1))
        past = [x.clone() for x in a.model.stacked_kv_cache]
        captured = [{} for _ in a.model.model.layers]
        hooks = []
        for i, layer in enumerate(a.model.model.layers):
            def pre(_, inputs, i=i):
                captured[i]['input'] = inputs[0].hidden_states.flatten().clone()
            def attention(_, inputs, output, i=i):
                captured[i]['oproj'] = output.hidden_states.flatten().clone()
            def full(_, inputs, output, i=i):
                captured[i]['downproj'] = output.hidden_states.flatten().clone()
            def attn_input(_, inputs, i=i):
                captured[i]['partial'] = inputs[0].flatten().clone()
            def down_input(_, inputs, i=i):
                captured[i]['upgate'] = inputs[0].flatten().clone()
            hooks += [layer.register_forward_pre_hook(pre), layer.self_attn.register_forward_hook(attention),
                      layer.register_forward_hook(full), layer.self_attn.o_proj.register_forward_pre_hook(attn_input),
                      layer.mlp.down_proj.register_forward_pre_hook(down_input)]
        a.context = args.context + args.step
        a.position.fill_(a.context)
        a.state.seq_len = a.context + 1
        a.run()
        ref_kv = [x[:, 0, a.context].clone() for x in a.model.stacked_kv_cache]
        for hook in hooks:
            hook.remove()
        a.engine = 'native'
        g = a.globs
        for layer, references in enumerate(captured):
            for op in ('partial', 'oproj', 'upgate', 'downproj'):
                for x, before in zip(a.model.stacked_kv_cache, past):
                    x.copy_(before)
                g.hidden_states.copy_(references['input'])
                g.barriers.zero_()
                if layer:
                    g.barriers[layer-1, 5, 0] = 512
                g.pos_id = a.context
                nodes, outputs = make_dag_layer(g, layer, [], op)
                schedule = Schedule(g, nodes, DAG_Node(NoOp(), outputs))
                tensorize_instructions(g, assign_to_sms(a.config.scheduler, schedule=schedule))
                a.interpret(g, a.native)
                out = {'partial': g.attn_out, 'oproj': g.hidden_states,
                       'upgate': g.silu_out, 'downproj': g.hidden_states}[op]
                row = {'layer': layer, 'op': op, 'output': errors(out, references[op]),
                       'k_cache': errors(g.k_cache[layer, 0, a.context], ref_kv[0][layer]),
                       'v_cache': errors(g.v_cache[layer, 0, a.context], ref_kv[1][layer])}
                report['rows'].append(row)
                write(args.output, report)
    print(json.dumps({'output': str(args.output), 'comparisons': len(report['rows'])}))


if __name__ == '__main__':
    main()
