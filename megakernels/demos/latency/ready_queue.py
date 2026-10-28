"""Host contracts for opt-in ready dispatch of the existing Llama instruction ISA."""
import heapq
from collections import Counter

from megakernels.scheduler import serialize_and_pad


def pack_dag(nodes):
    """Return canonical outgoing CSR/indegrees; reject malformed task graphs."""
    indices={node:i for i,node in enumerate(nodes)}
    if len(indices)!=len(nodes): raise ValueError('duplicate task node')
    children=[[] for _ in nodes];indegrees=[]
    for i,node in enumerate(nodes):
        deps=node.dependencies
        if len(set(deps))!=len(deps):raise ValueError('duplicate producer edge')
        if any(parent not in indices for parent in deps):raise ValueError('missing producer')
        indegrees.append(len(deps))
        for parent in deps:children[indices[parent]].append(i)
    pending=list(indegrees);ready=[i for i,k in enumerate(pending)if k==0];heapq.heapify(ready);count=0
    while ready:
        i=heapq.heappop(ready);count+=1
        for child in children[i]:
            pending[child]-=1
            if pending[child]==0:heapq.heappush(ready,child)
    if count!=len(nodes):raise ValueError('dependency cycle')
    offsets=[0];edges=[]
    for row in children:edges.extend(sorted(row));offsets.append(len(edges))
    n=len(nodes)
    if 3+3*n>=2**31 or len(edges)+2*n+1>=2**31:raise ValueError('ready metadata exceeds int32 indexing')
    instructions=[serialize_and_pad(node.instruction)for node in nodes]
    if any(not 0<=words[0]<=7 for words in instructions):raise ValueError('unsupported task opcode')
    return {'instructions':instructions,'offsets':offsets,'children':edges,'indegrees':indegrees,
            'metadata':offsets+edges+indegrees,'workspace_elements':3+3*n}


def audit_llama_fanins(schedule):
    """Cover the native skip-reduction Llama barrier producers, not just edges."""
    g=schedule.globs;nodes=schedule.dag_nodes
    if not g.skip_attn_reduction:raise ValueError('ready prototype requires skip-attention-reduction DAG')
    groups={}
    for node in nodes:
        op=node.instruction.opcode()
        if op not in (1,2,4,5,6,7):raise ValueError('unsupported model task')
        layer=getattr(node.instruction,'layer_idx',g.num_hidden_layers-1)
        if not 0<=layer<g.num_hidden_layers:raise ValueError('invalid task layer')
        groups.setdefault((layer,op),[]).append(node)
    def group(layer,op):return groups.get((layer,op),[])
    def cover(values,expected):
        counts=Counter(values)
        if set(counts)!=set(expected) or any(v!=1 for v in counts.values()):
            raise ValueError('missing or duplicate native barrier/output producer')
    def require(node,parents):
        if not set(parents)<=set(node.dependencies):raise ValueError('native barrier producer absent from DAG fan-in')
    for layer in range(g.num_hidden_layers):
        qkv=group(layer,1);partial=group(layer,2);oproj=group(layer,4);up=group(layer,5);down=group(layer,6)
        qblocks={}
        for node in qkv:
            ins=node.instruction
            for block in range(ins.start_output_block_idx,ins.end_output_block_idx):
                if block in qblocks:raise ValueError('duplicate QKV output')
                qblocks[block]=node
            if layer:require(node,group(layer-1,6))
        cover(qblocks.keys(),range((g.num_attention_heads+2*g.num_kv_heads)*g.head_dim//g.qkv_block_size))
        cover([node.instruction.kv_head_idx for node in partial],range(g.num_kv_heads))
        ratio=g.num_attention_heads//g.num_kv_heads
        if ratio!=4:raise ValueError('native attention prototype requires GQA ratio four')
        dims=g.head_dim//g.qkv_block_size
        for node in partial:
            ins=node.instruction
            if ins.num_partials!=1 or ins.partial_idx!=0:raise ValueError('unsupported attention partition')
            h=ins.kv_head_idx
            blocks=list(range(h*ratio*dims,(h+1)*ratio*dims))
            for head in (g.num_attention_heads+h,g.num_attention_heads+g.num_kv_heads+h):
                blocks.extend(range(head*dims,(head+1)*dims))
            require(node,{qblocks[b]for b in blocks})
        cover([b for node in oproj for b in range(node.instruction.start_block_idx,node.instruction.end_block_idx)],range(g.hidden_size//g.o_proj_block_size))
        for node in oproj:
            if node.instruction.reduction_block_idx!=0:raise ValueError('unexpected O projection reduction column')
            require(node,partial)
        cover([b for node in up for b in node.instruction.block_idxs],range(g.intermediate_size//g.up_gate_proj_block_size))
        for node in up:require(node,oproj)
        columns=g.intermediate_size//g.hidden_size
        cover([(node.instruction.reduction_block_idx,b)for node in down for b in range(node.instruction.start_block_idx,node.instruction.end_block_idx)],
              [(col,b)for col in range(columns)for b in range(g.hidden_size//g.down_proj_block_size)])
        for node in down:
            require(node,up)
            if getattr(g,'ordered_downproj',False):
                require(node,[n for n in down if n.instruction.reduction_block_idx==node.instruction.reduction_block_idx-1])
    head=group(g.num_hidden_layers-1,7)
    if any(group(layer,7)for layer in range(g.num_hidden_layers-1)):raise ValueError('early LM head')
    cover([b for node in head for b in range(node.instruction.start_output_block_idx,node.instruction.end_output_block_idx)],range(g.vocab_size//g.lm_head_block_size))
    for node in head:require(node,group(g.num_hidden_layers-1,6))


def tensorize_ready(schedule, *, audit_model=True):
    import torch
    if audit_model:audit_llama_fanins(schedule)
    packed=pack_dag(schedule.dag_nodes);g=schedule.globs;n=len(packed['instructions'])
    g.instructions=torch.tensor(packed['instructions'],dtype=torch.int32,device=g.device).reshape(1,n,32)
    g.timings=torch.zeros((g.sm_count(),n+1,128),dtype=torch.int32,device=g.device)
    g.ready_metadata=torch.tensor(packed['metadata'],dtype=torch.int32,device=g.device)
    g.ready_state=torch.empty(packed['workspace_elements'],dtype=torch.int32,device=g.device)
    return {'tasks':n,'edges':len(packed['children']), 'instruction_bytes':g.instructions.numel()*4,
            'metadata_bytes':g.ready_metadata.numel()*4,'workspace_bytes':g.ready_state.numel()*4,
            'timing_capacity_bytes':g.timings.numel()*4,'fanins_audited':audit_model}


def completion_check(g):
    n=g.instructions.shape[1];values=g.ready_state.cpu().tolist()
    head,tail,done=values[:3];slots=values[3:3+n];pending=values[3+n:3+2*n];visits=values[3+2*n:]
    passed=(head==tail==done==n and sorted(slots)==list(range(n)) and all(x==0 for x in pending) and all(x==1 for x in visits))
    return {'passed':passed,'tasks':n,'claimed':head,'published':tail,'completed':done,
            'all_tasks_visited_once':all(x==1 for x in visits),'all_dependencies_released':all(x==0 for x in pending),
            'all_task_ids_published_once':sorted(slots)==list(range(n))}
