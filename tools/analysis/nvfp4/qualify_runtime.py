"""Serialize sanitizer checks of TMEM reentry and dependent output publication."""
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'docs/thor/nvfp4'
def main():
    results=[]
    for variant in ('auto','2'):
        for tool in ('memcheck','synccheck','racecheck'):
            command=['compute-sanitizer','--tool',tool,'--kernel-name','kns=persistent_linear',
                     '--error-exitcode','7','--print-limit','5',sys.executable,
                     '-m','tools.analysis.nvfp4.runtime_bench','--check-only','--width','256','--length','3',
                     '--library',str(ROOT/f'local/thor-nvfp4/persistent-{variant}.so'),
                     '--output',str(OUT/f'persistent-check-{variant}-{tool}.json')]
            with (OUT/f'persistent-{variant}-{tool}.txt').open('w') as log:
                process=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=120)
            results.append(dict(variant=variant,tool=tool,returncode=process.returncode,command=command))
            (OUT/'persistent-sanitizers.json').write_text(json.dumps(results,indent=2)+'\n')
            print(variant,tool,process.returncode,flush=True)
            if process.returncode:raise SystemExit(process.returncode)
if __name__=='__main__':main()
