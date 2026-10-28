"""Serialized checks of native MMA inside the original VM worker/page loops."""
import json
import subprocess
import sys
from pathlib import Path
from .bench import ROOT
OUT=ROOT/'docs/thor/native-fp4-vm'
def run(name,flags,tool=None):
 command=[sys.executable,'-m','tools.analysis.nvfp4.vm_bench','--check-only',
          '--output',str(OUT/(name+'.json')),*flags]
 if tool:command=['compute-sanitizer','--tool',tool,'--kernel-name','kns=megakernel',
                  '--error-exitcode','7','--print-limit','5',*command]
 with (OUT/(name+'.txt')).open('w') as log:
  process=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=120)
 print(name,process.returncode,flush=True)
 return dict(name=name,command=command,returncode=process.returncode)
def main():
 OUT.mkdir(parents=True,exist_ok=True);results=[]
 fixtures=[('single',['--width','256','--length','1','--workers','1','--no-quant']),
           ('partial',['--width','512','--rows','264','--length','1','--workers','3','--noops']),
           ('ring',['--width','256','--length','17','--workers','20','--fused-quant','--noops']),
           ('dependent',['--width','2048','--length','8','--workers','20','--fused-quant'])]
 for name,flags in fixtures:
  results.append(run('check-'+name,flags))
  (OUT/'qualification.json').write_text(json.dumps(results,indent=2)+'\n')
  if results[-1]['returncode']:raise SystemExit(results[-1]['returncode'])
 for tool in ('memcheck','synccheck','racecheck'):
  for name,flags in fixtures[:3]:
   results.append(run(tool+'-'+name,flags,tool))
   (OUT/'qualification.json').write_text(json.dumps(results,indent=2)+'\n')
   if results[-1]['returncode']:raise SystemExit(results[-1]['returncode'])
if __name__=='__main__':main()
