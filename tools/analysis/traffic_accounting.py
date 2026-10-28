"""Count source-level requested bytes from the unchanged native execution plan.

Counts are not DRAM measurements. Polling/cache-line/atomic transaction traffic
cannot be inferred from the instruction plan and is explicitly excluded.
"""
import argparse,hashlib,json,math
from collections import Counter,defaultdict
from pathlib import Path
import torch
from tools.thor.model_bench import ModelAdapter

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--context',type=int,default=1024);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 with torch.inference_mode():
  a=ModelAdapter('native',args.context,1);g=a.globs;h=g.hidden_size;d=g.intermediate_size;layers=g.num_hidden_layers;counts=Counter();byte_counts=Counter();coverage=defaultdict(Counter);counter_rmws=0
  for node in a.schedule.dag_nodes:
   i=node.instruction;op=i.opcode();counts[op]+=1;layer=getattr(i,'layer_idx',-1)
   if op in (1,7):blocks=list(range(i.start_output_block_idx,i.end_output_block_idx))
   elif op in (4,6):blocks=list(range(i.start_block_idx,i.end_block_idx))
   elif op==5:blocks=i.block_idxs
   else:blocks=[]
   if op in (1,4,5,6,7):
    multiplier=2 if op==5 else 1
    byte_counts['linear_weight_reads']+=len(blocks)*multiplier*16*h*2
    byte_counts['activation_reads']+=h*2
    if op in (1,5,7):byte_counts['normalization_weight_reads']+=h*2
    for block in blocks:coverage[(op,layer,getattr(i,'reduction_block_idx',0))][block]+=1
   if op==1:
    byte_counts['rope_table_reads']+=g.head_dim*4*2
    byte_counts['qkv_writes']+=len(blocks)*16*2;counter_rmws+=len(blocks)
   elif op==2:
    assert i.num_partials==1 and g.skip_attn_reduction
    tiles=math.ceil((args.context+1)/g.attn_kv_block_size)
    byte_counts['kv_reads_requested_with_tail_padding']+=2*tiles*g.attn_kv_block_size*g.head_dim*2
    byte_counts['attention_query_reads']+=4*g.head_dim*2
    byte_counts['attention_output_writes']+=4*g.head_dim*2;counter_rmws+=4
   elif op in (4,6):
    byte_counts['residual_RMW_reads']+=len(blocks)*16*2
    byte_counts['residual_RMW_writes']+=len(blocks)*16*2;counter_rmws+=1
   elif op==5:byte_counts['silu_output_writes']+=len(blocks)*16*2;counter_rmws+=len(blocks)
   elif op==7:byte_counts['logit_writes']+=len(blocks)*16*2
  # Exact coverage of all physical linear-weight tiles, including all four
  # down-projection columns, is required before a compulsory/read-once claim.
  expected={}
  for layer in range(layers):
   expected[(1,layer,0)]=(g.num_attention_heads+2*g.num_kv_heads)*g.head_dim//16
   expected[(4,layer,0)]=h//16;expected[(5,layer,0)]=d//16
   for col in range(d//h):expected[(6,layer,col)]=h//16
  expected[(7,-1,0)]=g.vocab_size//16
  assert set(expected)==set(coverage)
  for key,n in expected.items():assert coverage[key]==Counter(range(n)),key
  linear_unique=sum(t.numel()*t.element_size()for t in [g.qkv_proj_weights,g.o_proj_weights,g.up_proj_weights,g.gate_proj_weights,g.down_proj_weights,g.lm_head_weights])
  assert byte_counts['linear_weight_reads']==linear_unique
  byte_counts['instruction_fetch_reads']=g.instructions.numel()*g.instructions.element_size()
  unique_norm=(2*layers+1)*h*2
  unique_activation=(layers*(3*h+d)+h)*2
  result={'status':'PASS','scope':'one native VM decode, source-level requested payload bytes; no DRAM/cache-line inference','context':args.context,'valid_kv_tokens':args.context+1,'tasks':sum(counts.values()),'instructions_by_opcode':dict(counts),'bytes':dict(byte_counts),'total_accounted_payload_bytes':sum(byte_counts.values()),'linear_weight_fraction':byte_counts['linear_weight_reads']/sum(byte_counts.values()),'linear_weight_coverage':'every linear weight tile appears exactly once','normalization_unique_bytes':unique_norm,'normalization_repeated_requested_bytes':byte_counts['normalization_weight_reads']-unique_norm,'activation_unique_stage_input_bytes':unique_activation,'activation_repeated_requested_bytes':byte_counts['activation_reads']-unique_activation,'counter_RMW_operations':counter_rmws,'outside_VM':{'barrier_reset_writes':g.barriers.numel()*4,'embedding_lookup_payload_reads':h*2,'embedding_output_write_and_copy_payload_bytes':h*2*3,'argmax_minimum_logit_reads':g.vocab_size*2},'not_counted':['number of global polling iterations','cache-line amplification/cache hits/evictions','atomic transaction amplification','instruction cache fetches and spills','shared-memory/register traffic','CPU/other-engine memory traffic'],'metadata':a.metadata,'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k in ['status','tasks','total_accounted_payload_bytes','linear_weight_fraction','activation_repeated_requested_bytes','normalization_repeated_requested_bytes']}))
if __name__=='__main__':main()
