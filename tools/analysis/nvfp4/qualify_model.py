"""Check the complete FP4 decoder, including the reused attention operators."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from .bench import ROOT

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--weight-only',action='store_true');ap.add_argument('--output',type=Path,default=ROOT/'docs/thor/native-fp4-vm');args=ap.parse_args()
 out=args.output;out.mkdir(parents=True,exist_ok=True);records=[]
 for tool in ('memcheck','synccheck','racecheck'):
  command=['compute-sanitizer','--tool',tool,'--kernel-name','kns=megakernel',
   '--error-exitcode','7','--print-limit','3',sys.executable,'-m','tools.analysis.nvfp4.model_bench',
   '--check-only','--context','1024','--output',str(out/f'model-{tool}.json')]
  if args.weight_only:command.append('--weight-only')
  with (out/f'model-{tool}.txt').open('w') as log:
   p=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=240)
  records.append(dict(tool=tool,returncode=p.returncode,command=command))
  (out/'model-sanitizers.json').write_text(json.dumps(records,indent=2)+'\n')
  print(tool,p.returncode,flush=True)
  if p.returncode:raise SystemExit(p.returncode)
if __name__=='__main__':main()
