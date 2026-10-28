"""Apply scoped observer-only timing to an isolated diagnostic worktree."""
import argparse
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('worktree',type=Path);args=p.parse_args();w=args.worktree.resolve()
 if w==Path(__file__).resolve().parents[2] or not (w/'.git').is_file():
  raise RuntimeError('requires an isolated linked worktree; refusing to patch the production checkout')
 def edit(name,old,new,count=1):
  p=w/name;s=p.read_text()
  if s.count(old)!=count:raise RuntimeError(f'{name}: expected {count} occurrences, got {s.count(old)}')
  p.write_text(s.replace(old,new))
 edit('demos/low-latency-llama/llama.cuh','    timing_layout timings;','    timing_layout timings;\n    kittens::gl<int, 1, -1, -1, 12> diag;')
 edit('demos/low-latency-llama/llama.cuh','template <int _num_layers,', '''// Observer clocks are local to one SM. Rows are per worker/instruction;
// sampled consumer warp 0 and loader lane 0 write disjoint columns.
__device__ inline unsigned diag_clock() {
    unsigned value; asm volatile("mov.u32 %0, %%clock;" : "=r"(value) :: "memory"); return value;
}
template<typename G,typename S> __device__ inline void diag_write(const G& g,const S& s,int field,unsigned cycles) {
    g.diag.raw_ptr[(megakernel::get_worker_id()*g.instructions.rows()+s.instruction_index)*12+field]=static_cast<int>(cycles);
}

template <int _num_layers,''')
 edit('demos/low-latency-llama/llama.cu','&llama_1b_globals::timings,','&llama_1b_globals::timings, &llama_1b_globals::diag,')
 edit('megakernels/demos/latency/mk.py','        globs.timings,','        globs.timings,\n        globs.diag,')
 edit('tools/thor/model_bench.py',"        if engine == 'native':", "        if self.globs is not None:\n            self.globs.diag = torch.zeros((*self.globs.instructions.shape[:2],12),device='cuda',dtype=torch.int32)\n        if engine == 'native':")
 p=w/'demos/low-latency-llama/matvec_pipeline.cuh';s=p.read_text()
 old='''        if (kittens::laneid() == 0) {

            int input_stage = 0;''';new='''        if (kittens::laneid() == 0) {
            unsigned diag_begin=diag_clock(), diag_reuse=0, diag_pages=0;
            int input_stage = 0;''';assert s.count(old)==1;s=s.replace(old,new)
 old='''                kittens::wait(weights_finished(s, input_stage),
                     (iter % (2 * INPUT_PIPELINE_STAGES)) <
                         INPUT_PIPELINE_STAGES);''';new='''                unsigned diag_wait=diag_clock();
'''+old+'''
                diag_reuse += diag_clock()-diag_wait;''';assert old in s;s=s.replace(old,new)
 old='''                        s.wait_page_ready(weight_page);''';new='''                        unsigned diag_page=diag_clock();
                        s.wait_page_ready(weight_page);
                        diag_pages += diag_clock()-diag_page;''';assert s.count(old)==1;s=s.replace(old,new)
 old='''        } else if (kittens::laneid() >= needed_pages''';new='''            diag_write(g,s,4,diag_reuse);diag_write(g,s,5,diag_pages);diag_write(g,s,6,diag_clock()-diag_begin);
        } else if (kittens::laneid() >= needed_pages''';assert old in s;s=s.replace(old,new)
 old='''        int input_stage = 0, output_stage = 0;
        for (int i = 0; i < inst.iters; i++) {''';new='''        const bool diag_sample=kittens::warpid()==0&&kittens::laneid()==0;
        unsigned diag_begin=0,diag_weight=0,diag_output=0,diag_work=0,diag_tick=0;
        if(diag_sample)diag_begin=diag_clock();
        int input_stage = 0, output_stage = 0;
        for (int i = 0; i < inst.iters; i++) {''';assert old in s;s=s.replace(old,new)
 old='''            kittens::wait(weights_arrived(s, input_stage),
                 (i % (2 * INPUT_PIPELINE_STAGES)) >= INPUT_PIPELINE_STAGES);''';new='''            if(diag_sample)diag_tick=diag_clock();
'''+old+'''
            if(diag_sample){diag_weight+=diag_clock()-diag_tick;diag_tick=diag_clock();}''';assert old in s;s=s.replace(old,new)
 old='''            kittens::wait(outputs_finished(s, output_stage),
                 (i % (2 * OUTPUT_PIPELINE_STAGES)) < OUTPUT_PIPELINE_STAGES);''';new=old+'''
            if(diag_sample)diag_output+=diag_clock()-diag_tick;''';assert old in s;s=s.replace(old,new)
 old='''            matvec(out_smem, weights, activations_vec);''';new='''            if(diag_sample)diag_tick=diag_clock();
'''+old+'''
            if(diag_sample)diag_work+=diag_clock()-diag_tick;''';assert old in s;s=s.replace(old,new)
 old='''    template <int iter_scale = 1>''';idx=s.index(old);end=s.rindex('        }\n    }\n\n',0,idx);s=s[:end]+s[end:].replace('''        }
    }

''','''        }
        if(diag_sample){diag_write(g,s,0,diag_weight);diag_write(g,s,1,diag_output);diag_write(g,s,2,diag_work);diag_write(g,s,3,diag_clock()-diag_begin);diag_write(g,s,10,s.instruction()[0]);diag_write(g,s,11,inst.iters);}
    }

''',1)
 old='''            pipeline_specifics::gmem_wait(g, s);''';assert s.count(old)==1;s=s.replace(old,'''            unsigned diag_dep=diag_clock();
            pipeline_specifics::gmem_wait(g, s);
            diag_write(g,s,7,diag_clock()-diag_dep);''')
 old='''        using sv_t = kittens::sv_bf<REDUCTION_DIM_PER_WARP>;''';assert s.count(old)==1;s=s.replace(old,old+'''
        unsigned diag_setup=0;if(kittens::warpid()==0&&kittens::laneid()==0)diag_setup=diag_clock();''')
 old='''        pipeline::consumer_loop(s, g, activations_vec);''';assert s.count(old)==1;s=s.replace(old,'''        if(kittens::warpid()==0&&kittens::laneid()==0)diag_write(g,s,8,diag_clock()-diag_setup);
'''+old);p.write_text(s)
 p=w/'demos/low-latency-llama/matvec_adds.cu';s=p.read_text()
 old='''            using sv_t = kittens::sv_bf<pipeline::REDUCTION_DIM_PER_WARP>;''';assert old in s;s=s.replace(old,old+'''
            unsigned diag_setup=0;if(kittens::warpid()==0&&kittens::laneid()==0)diag_setup=diag_clock();''')
 old='''                megakernel::gmem_wait<Config>([&] { return (gmem_load_acquire''';assert s.count(old)==1;s=s.replace(old,'''                unsigned diag_dep=diag_clock();
'''+old)
 old='''                       EXPECTED_ARRIVAL_COUNT); });''';assert old in s;s=s.replace(old,old+'''
                diag_write(g,s,7,diag_clock()-diag_dep);''')
 old='''            pipeline::consumer_loop(s, g, activations_vec);''';assert old in s;s=s.replace(old,'''            if(kittens::warpid()==0&&kittens::laneid()==0)diag_write(g,s,8,diag_clock()-diag_setup);
'''+old);p.write_text(s)
 print(w)
if __name__=='__main__':main()
