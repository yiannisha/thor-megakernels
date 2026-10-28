"""Held-out decode likelihood comparison, BF16-prefilled fixed-context windows.

This is a deterministic subset, not the standard whole-corpus perplexity score.
"""
import argparse,hashlib,json,math
from pathlib import Path
import torch
from transformers import AutoTokenizer
from megakernels.model_types import BatchState
from .model_bench import FP4ModelAdapter
from .bench import ROOT
@torch.inference_mode()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--windows',type=int,default=16);ap.add_argument('--prefix',type=int,default=128);ap.add_argument('--suffix',type=int,default=64);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 source=ROOT/'local/thor-native-fp4-vm/wikitext-test-rows.json';raw=source.read_bytes();data=json.loads(raw)
 tokenizer=AutoTokenizer.from_pretrained(ROOT/'local/models/Llama-3.2-1B-Instruct',local_files_only=True)
 text='\n\n'.join(r['row']['text'] for r in data['rows'] if r['row']['text'].strip())
 ids=tokenizer.encode(text,add_special_tokens=False)
 a=FP4ModelAdapter();cache=a.model.stacked_kv_cache
 def restore(saved):
  for dst,src in zip(cache,saved):dst.copy_(src)
 def step(mode,pos,token):
  a.context=pos;a.position.fill_(pos);a.state.seq_len=pos+1;a.case.token.fill_(token)
  return a._run_torch() if mode=='bf16' else a.run()
 result={'dataset':'Salesforce/wikitext','config':'wikitext-2-raw-v1','split':'test',
  'source_sha256':hashlib.sha256(raw).hexdigest(),'prefix_tokens':args.prefix,'suffix_tokens':args.suffix,
  'scope':'BF16 prefill for both; distinct teacher-forced BF16/FP4 decode caches; deterministic first windows; not full-corpus PPL','windows':[]}
 def write():args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
 for window in range(args.windows):
  start=window*(args.prefix+args.suffix);chunk=ids[start:start+args.prefix+args.suffix]
  if len(chunk)<args.prefix+args.suffix:raise ValueError('insufficient source tokens')
  prefix=torch.tensor([chunk[:args.prefix-1]],device='cuda');positions=torch.arange(prefix.shape[1],device='cuda').unsqueeze(0)
  for v in cache:v.zero_()
  a.model(BatchState(input_ids=prefix,position_ids=positions,seq_len=prefix.shape[1]))
  states={mode:[v.clone() for v in cache] for mode in ('bf16','fp4')};rows=[];token=chunk[args.prefix-1]
  for i,target in enumerate(chunk[args.prefix:]):
   lp={};top={}
   for mode in ('bf16','fp4'):
    restore(states[mode]);logits=step(mode,args.prefix-1+i,token).float();lp[mode]=logits.log_softmax(-1);top[mode]=int(logits.argmax())
    for dst,src in zip(states[mode],cache):dst.copy_(src)
   rows.append(dict(bf16_nll=float(-lp['bf16'][target]),fp4_nll=float(-lp['fp4'][target]),
    kl=float((lp['bf16'].exp()*(lp['bf16']-lp['fp4'])).sum()),top1_agrees=top['bf16']==top['fp4']))
   token=target
  result['windows'].append(dict(index=window,source_token_offset=start,metrics=rows));write();print('window',window,'FP4/BF16 loss',sum(x['fp4_nll'] for x in rows)/len(rows),sum(x['bf16_nll'] for x in rows)/len(rows),flush=True)
 rows=[r for w in result['windows'] for r in w['metrics']];b=sum(r['bf16_nll'] for r in rows)/len(rows);f=sum(r['fp4_nll'] for r in rows)/len(rows)
 result['summary']=dict(tokens=len(rows),bf16_nll=b,fp4_nll=f,bf16_conditional_ppl=math.exp(b),fp4_conditional_ppl=math.exp(f),perplexity_ratio=math.exp(f-b),
  mean_kl=sum(r['kl'] for r in rows)/len(rows),top1_agreement=sum(r['top1_agrees'] for r in rows)/len(rows));write();print(result['summary'],flush=True)
if __name__=='__main__':main()
