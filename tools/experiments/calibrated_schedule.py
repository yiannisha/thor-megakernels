"""Opt-in static DAG costs fitted to qualified same-warp native timing spans.

THOR_COST_PROFILE selects a raw instruction-profile JSON. The model/runtime,
context and numerical/resource configuration must match. Timing and scheduler
selection may differ. This is an experimental inclusive-service cost model;
spans can include dependency waits and do not isolate arithmetic execution.
"""
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import statistics


def cost_key(opcode, default_cost):
    if not math.isfinite(default_cost) or default_cost <= 0:
        raise ValueError('invalid base cost')
    return f'{opcode}:{float(default_cost):.17g}'


def signature(metadata):
    config = {k:v for k,v in metadata['config'].items() if k not in ('name','timing','scheduler')}
    return {k:metadata[k] for k in ('model','model_provenance','precision','context','batch')} | {
        'config':config, 'source_hash':metadata['native_build']['source_hash'],
        'reference_attention':metadata.get('reference_attention'),
        'prefill_attention':metadata.get('prefill_attention'),
        'float32_matmul_precision':metadata.get('float32_matmul_precision'),
        'gate':metadata.get('gate')}


def fit_profile(profile):
    if profile.get('schema_version') != 'thor.instruction-profile.v1' or profile.get('status') != 'PASS':
        raise ValueError('a passing native instruction profile is required')
    checks = profile.get('checks',[])
    if not checks or not all(c.get('passed') is True for c in checks):
        raise ValueError('profile numerical checks missing or failed')
    groups = {}
    for row in profile['rows']:
        value = row['consumer_max_cycles']
        if not math.isfinite(value) or value <= 0:
            raise ValueError('invalid measured cost')
        groups.setdefault(cost_key(row['opcode'],row['default_cost']),[]).append(value)
    if not groups:
        raise ValueError('empty instruction profile')
    return {key:{'median_cycles':statistics.median(values),'samples':len(values),
                 'min_cycles':min(values),'max_cycles':max(values)} for key,values in groups.items()}


def assign_critical_path(schedule, locality=False):
    """Ready-only static list scheduling; optional producer-affinity tie break.

    Affinity only breaks equal estimated completion times. It does not retain
    activations in registers/shared memory or promise lower DRAM traffic.
    """
    from megakernels.scheduler import validate_queues
    nodes = schedule.dag_nodes
    indices = {node:i for i,node in enumerate(nodes)}
    if len(indices) != len(nodes):
        raise ValueError('duplicate DAG node')
    children = {node:[] for node in nodes}
    pending = {node:len(set(node.dependencies)) for node in nodes}
    for node in nodes:
        for parent in set(node.dependencies):
            if parent not in indices: raise ValueError('missing producer')
            children[parent].append(node)
    ready = [indices[n] for n in nodes if pending[n]==0]; heapq.heapify(ready)
    order=[]
    while ready:
        node=nodes[heapq.heappop(ready)]; order.append(node)
        for child in children[node]:
            pending[child]-=1
            if pending[child]==0:heapq.heappush(ready,indices[child])
    if len(order)!=len(nodes):raise ValueError('dependency cycle')
    costs={node:schedule.instruction_cost(node.instruction) for node in nodes}
    critical={}
    for node in reversed(order):
        critical[node]=costs[node]+max((critical[c] for c in children[node]),default=0)
    workers=schedule.globs.sm_count()
    if workers<1:raise ValueError('positive worker count required')
    queues=[[] for _ in range(workers)]; available=[0]*workers
    pending={node:len(set(node.dependencies)) for node in nodes}
    ready=[(-critical[n],indices[n]) for n in nodes if pending[n]==0];heapq.heapify(ready)
    placement={};ends={}
    while ready:
        _,index=heapq.heappop(ready);node=nodes[index]
        release=max((ends[p] for p in node.dependencies),default=0)
        def score(worker):
            finish=max(available[worker],release)+costs[node]
            affinity=sum(placement[p]==worker for p in set(node.dependencies)) if locality else 0
            return finish,-affinity,worker
        worker=min(range(workers),key=score)
        node.start_time=max(available[worker],release)
        node.end_time=node.start_time+costs[node]
        available[worker]=node.end_time;ends[node]=node.end_time;placement[node]=worker
        queues[worker].append(node.instruction)
        for child in children[node]:
            pending[child]-=1
            if pending[child]==0:heapq.heappush(ready,(-critical[child],indices[child]))
    validate_queues(schedule,queues)
    return queues


def make_adapter():
    from tools.thor.model_bench import ModelAdapter
    from megakernels.scheduler import Schedule, assign_to_sms, tensorize_instructions
    adapter = ModelAdapter('native',int(os.environ.get('THOR_CONTEXT','1024')),int(os.environ.get('THOR_SEED','1')))
    if adapter.config.scheduler not in ('dag','wave'):
        raise ValueError('calibrated adapter requires dag or wave scheduler')
    path = Path(os.environ['THOR_COST_PROFILE'])
    raw = path.read_bytes(); profile = json.loads(raw)
    if signature(profile['metadata']) != signature(adapter.metadata):
        raise ValueError('profile model, source, context or configuration mismatch')
    costs = fit_profile(profile)
    missing = set()
    def model(instruction, globs):
        key = cost_key(instruction.opcode(),instruction.cost(globs))
        if key not in costs:
            missing.add(key)
            raise ValueError(f'profile missing instruction shape {key}')
        return costs[key]['median_cycles']
    schedule = Schedule(adapter.globs,adapter.schedule.dag_nodes,adapter.schedule.end_node,cost_model=model)
    policy=os.environ.get('THOR_COST_POLICY','greedy')
    if policy not in ('greedy','critical','locality'):
        raise ValueError('unknown calibrated scheduling policy')
    queues = (assign_to_sms(adapter.config.scheduler,schedule=schedule) if policy=='greedy'
              else assign_critical_path(schedule,locality=policy=='locality'))
    tensorize_instructions(adapter.globs,queues)
    adapter.schedule = schedule
    queue_bytes = adapter.globs.instructions.cpu().numpy().tobytes()
    adapter.metadata.update(cost_model={'type':'inclusive-consumer-max-median-cycles',
                            'queue_policy':policy,'profile_path':str(path.resolve()),'profile_sha256':hashlib.sha256(raw).hexdigest(),
                            'profile_signature':signature(profile['metadata']),'buckets':costs,
                            'queue_sha256':hashlib.sha256(queue_bytes).hexdigest(),
                            'limitations':profile['limitations']},
                            experiment_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return adapter
