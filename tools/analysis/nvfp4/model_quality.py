"""Small natural-language sanity set, not a general quality benchmark.

Both paths use the same BF16 prefill. Decode caches are separate. Teacher-forced
continuations measure token likelihood; greedy text exposes behavioral changes.
"""
import argparse,json,math
from pathlib import Path
import torch
from transformers import AutoTokenizer
from megakernels.model_types import BatchState
from .model_bench import FP4ModelAdapter
from .bench import ROOT
CASES=[
 ('What is 7 + 5? Answer with only the number.','12'),
 ('What is the capital of France? Answer with only the city name.','Paris'),
 ("Translate 'good morning' into French. Answer with just the translation.",'Bonjour'),
 ('Complete the sequence: 2, 4, 6, 8, ... Answer with only the next number.','10'),
 ('Name the largest planet in our solar system. Answer with only its name.','Jupiter'),
 ('A box contains 3 red balls and 2 blue balls. How many balls are in the box? Answer with only the number.','5'),
 ('Choose the correct word: She ___ to school every day. (go/goes) Answer with one word.','goes'),
 ('Explain in one sentence why ice floats on water.','Ice is less dense than liquid water.'),
]
@torch.inference_mode()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--cases',type=int,default=8);ap.add_argument('--max-new-tokens',type=int,default=48);ap.add_argument('--weight-only',action='store_true');ap.add_argument('--mixed-fp8',action='store_true');args=ap.parse_args()
 a=FP4ModelAdapter(weight_only=args.weight_only,mixed_fp8=args.mixed_fp8);tokenizer=AutoTokenizer.from_pretrained(ROOT/'local/models/Llama-3.2-1B-Instruct',local_files_only=True)
 cache=a.model.stacked_kv_cache
 def clone():return [v.clone() for v in cache]
 def restore(saved):
  for dst,src in zip(cache,saved):dst.copy_(src)
 def step(mode,position,token):
  a.context=position;a.position.fill_(position);a.state.seq_len=position+1;a.case.token.fill_(token)
  return a._run_torch() if mode=='bf16' else a.run()
 result={'precision':'W4A8' if args.mixed_fp8 else 'W4A16' if args.weight_only else 'W4A4','scope':'8 authored sanity prompts; BF16 prefill; separate autoregressive decode caches; no latency claims', 'cases':[]}
 stop={tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids('<|eot_id|>')}
 def write():args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
 for prompt,answer in CASES[:args.cases]:
  ids=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True)
  prefix=torch.tensor([ids[:-1]],device='cuda')
  for v in cache:v.zero_()
  positions=torch.arange(prefix.shape[1],device='cuda').unsqueeze(0)
  a.model(BatchState(input_ids=prefix,position_ids=positions,seq_len=prefix.shape[1]))
  initial=clone();states={'bf16':clone(),'fp4':clone()};targets=tokenizer.encode(answer,add_special_tokens=False)
  row={'prompt':prompt,'target':answer,'prefix_tokens':len(ids),'teacher_forced':[],'greedy':{}}
  token=ids[-1]
  for i,target in enumerate(targets):
   logps={};tops={}
   for mode in ('bf16','fp4'):
    restore(states[mode]);logits=step(mode,len(ids)-1+i,token).float();logps[mode]=logits.log_softmax(-1);tops[mode]=int(logits.argmax());states[mode]=clone()
   row['teacher_forced'].append(dict(target_id=target,bf16_nll=float(-logps['bf16'][target]),fp4_nll=float(-logps['fp4'][target]),
    kl_bf16_to_fp4=float((logps['bf16'].exp()*(logps['bf16']-logps['fp4'])).sum()),bf16_top1=tops['bf16'],fp4_top1=tops['fp4']))
   token=target
  for mode in ('bf16','fp4'):
   restore(initial);token=ids[-1];generated=[]
   for i in range(args.max_new_tokens):
    logits=step(mode,len(ids)-1+i,token);token=int(logits.argmax());generated.append(token)
    if token in stop:break
   row['greedy'][mode]={'token_ids':generated,'text':tokenizer.decode(generated,skip_special_tokens=True)}
  result['cases'].append(row);write();print(prompt,row['greedy'],flush=True)
 tokens=[t for c in result['cases'] for t in c['teacher_forced']]
 result['summary']={'cases':len(result['cases']),'teacher_forced_tokens':len(tokens),
  'mean_bf16_nll':sum(t['bf16_nll'] for t in tokens)/len(tokens),'mean_fp4_nll':sum(t['fp4_nll'] for t in tokens)/len(tokens),
  'mean_kl_bf16_to_fp4':sum(t['kl_bf16_to_fp4'] for t in tokens)/len(tokens),
  'top1_agreement':sum(t['bf16_top1']==t['fp4_top1'] for t in tokens)/len(tokens)}
 result['summary']['continuation_perplexity_ratio']=math.exp(result['summary']['mean_fp4_nll']-result['summary']['mean_bf16_nll']);write()
if __name__=='__main__':main()
