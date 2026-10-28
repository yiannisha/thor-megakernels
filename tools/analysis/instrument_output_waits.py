"""Extend the existing isolated wait observer with storer phase timing.
Fields 12..17: arrival wait, store call, slot publication, loop, TMA completion,
global readiness publication. Last two are nested within store call.
"""
import argparse
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('worktree',type=Path);a=p.parse_args();w=a.worktree.resolve()
assert (w/'.git').is_file() and w!=Path(__file__).resolve().parents[2]
def edit(name,old,new,count=1):
 f=w/name;s=f.read_text();assert s.count(old)==count,(name,old,s.count(old));f.write_text(s.replace(old,new))
edit('demos/low-latency-llama/llama.cuh','-1, -1, 12> diag','-1, -1, 18> diag')
edit('demos/low-latency-llama/llama.cuh','*12+field','*18+field')
edit('tools/thor/model_bench.py','shape[:2],12)','shape[:2],18)')
f='demos/low-latency-llama/matvec_pipeline.cuh'
edit(f,'        int output_stage = 0;\n        for (int i = 0; i < inst.iters; i++) {','''        const bool diag_sample=kittens::laneid()==0;
        unsigned diag_start=0,diag_arrival=0,diag_store=0,diag_publish=0,diag_tick=0,diag_tma=0,diag_global=0;
        if(diag_sample)diag_start=diag_clock();
        int output_stage = 0;
        for (int i = 0; i < inst.iters; i++) {''')
edit(f,'            kittens::wait(sem, bit);','''            if(diag_sample)diag_tick=diag_clock();
            kittens::wait(sem, bit);
            if(diag_sample){diag_arrival+=diag_clock()-diag_tick;diag_tick=diag_clock();}''')
edit(f,'            pipeline_specifics::store(s, g, inst, i, output_stage);','''            pipeline_specifics::store(s, g, inst, i, output_stage,diag_tma,diag_global);
            if(diag_sample){diag_store+=diag_clock()-diag_tick;diag_tick=diag_clock();}''')
edit(f,'            output_stage = (output_stage + 1) % OUTPUT_PIPELINE_STAGES;\n        }\n    }\n};','''            if(diag_sample)diag_publish+=diag_clock()-diag_tick;
            output_stage = (output_stage + 1) % OUTPUT_PIPELINE_STAGES;
        }
        if(diag_sample){diag_write(g,s,12,diag_arrival);diag_write(g,s,13,diag_store);diag_write(g,s,14,diag_publish);diag_write(g,s,15,diag_clock()-diag_start);diag_write(g,s,16,diag_tma);diag_write(g,s,17,diag_global);}
    }
};''')
for name in ['matvec_adds.cu','rms_lm_head.cu','rms_matvec_rope_append.cu','upgate.cu']:
 f=w/'demos/low-latency-llama'/name;s=f.read_text();old='int output_idx, int output_stage) {';assert s.count(old)==1;s=s.replace(old,'int output_idx, int output_stage, unsigned &diag_tma, unsigned &diag_global) {')
 import re
 before,s=s.split('int output_idx, int output_stage, unsigned &diag_tma, unsigned &diag_global) {',1)
 body,after=s.split('    using pipeline',1) if '    using pipeline' in s else (s,'')
 s=body
 s,n=re.subn(r'(                kittens::tma::store_async_(?:read_)?wait\(\);)',r'                unsigned diag_wait_start=diag_clock();\n\1\n                diag_tma+=diag_clock()-diag_wait_start;',s);assert n==1,(name,n)
 # Readiness atomics are nested within lane-zero branch.
 s,n=re.subn(r'(                (?:gmem_add_release\([\s\S]*?;))',r'                unsigned diag_global_start=diag_clock();\n\1\n                diag_global+=diag_clock()-diag_global_start;',s)
 assert n==(0 if name in ('matvec_adds.cu','rms_lm_head.cu') else 1),(name,n)
 f.write_text(before+'int output_idx, int output_stage, unsigned &diag_tma, unsigned &diag_global) {'+s+('    using pipeline'+after if after else ''))
print(w)
