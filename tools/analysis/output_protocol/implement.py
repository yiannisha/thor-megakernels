"""Apply real completion protocol variants to an isolated clean worktree."""
from pathlib import Path
import sys
w=Path(sys.argv[1]).resolve();assert (w/'.git').is_file()
def edit(name,old,new,n=1):
 p=w/name;s=p.read_text();assert s.count(old)==n,(name,old,s.count(old));p.write_text(s.replace(old,new))
edit('include/gmem_barrier.cuh','#include <stdint.h>','''#include <stdint.h>
#ifndef MK_PUBLICATION_RELEASE_ONLY
#define MK_PUBLICATION_RELEASE_ONLY 0
#endif''')
edit('include/gmem_barrier.cuh','cuda::memory_order_acq_rel);','''
#if MK_PUBLICATION_RELEASE_ONLY
        // Every producer is a release RMW on the same counter. Later RMWs
        // remain in earlier producers' release sequences; a final acquire
        // observing the target therefore sees every producer's stores.
        cuda::memory_order_release);
#else
        cuda::memory_order_acq_rel);
#endif''')
(w/'include/output_publication.cuh').write_text('''#pragma once
#include "gmem_barrier.cuh"
#ifndef MK_EARLY_OUTPUT_RELEASE
#define MK_EARLY_OUTPUT_RELEASE 0
#endif
#ifndef MK_ASYNC_PUBLICATION
#define MK_ASYNC_PUBLICATION 0
#endif
namespace megakernel {
// Per-instruction SPSC queue, storer -> launcher. This scratch interval is
// reserved between the reduction/RMS region and the RoPE region.
struct publication_record { unsigned *counter; int value; int reserved; };
struct publication_ring { publication_record records[8]; int head; int tail; };
template<class C> __device__ publication_ring &publication_queue(state<C>&s) {
 static_assert(C::SCRATCH_BYTES>=3584);
 return *reinterpret_cast<publication_ring*>(static_cast<char*>(s.scratch())+3200);
}
__device__ inline int queue_acquire(int &v) {
 return cuda::atomic_ref<int,cuda::thread_scope_block>(v).load(cuda::memory_order_acquire);
}
__device__ inline void queue_release(int &v,int n) {
 cuda::atomic_ref<int,cuda::thread_scope_block>(v).store(n,cuda::memory_order_release);
}
template<class C> __device__ void publication_init(state<C>&s) {
#if MK_ASYNC_PUBLICATION
 auto&q=publication_queue(s);q.head=0;q.tail=0;
#endif
}
template<class C> __device__ void publish_output(state<C>&s,unsigned*counter,int value) {
#if MK_ASYNC_PUBLICATION
 auto&q=publication_queue(s);int tail=q.tail;
 while(tail-queue_acquire(q.head)>=8)__nanosleep(20);
 q.records[tail%8]={counter,value,0};queue_release(q.tail,tail+1);
#else
 gmem_add_release(counter,value);
#endif
}
template<class C> __device__ void publication_drain(state<C>&s) {
#if MK_ASYNC_PUBLICATION
 if(kittens::laneid()==0){auto&q=publication_queue(s);int tail=q.tail;
 while(queue_acquire(q.head)!=tail)__nanosleep(20);}
 kittens::warp::sync();
#endif
}
template<class C> __device__ void publication_run(state<C>&s,int count) {
#if MK_ASYNC_PUBLICATION
 if(kittens::laneid()==0){auto&q=publication_queue(s);
 for(int head=0;head<count;head++){
 while(queue_acquire(q.tail)<=head)__nanosleep(20);
 auto record=q.records[head%8];
 gmem_add_release(record.counter,record.value);
 queue_release(q.head,head+1);
 }}
#endif
}
}
''')
edit('demos/low-latency-llama/matvec_pipeline.cuh','#include "llama.cuh"','#include "llama.cuh"\n#include "output_publication.cuh"')
edit('demos/low-latency-llama/matvec_pipeline.cuh','        init_semaphore(activations_arrived(s), 1);','        megakernel::publication_init(s);\n        init_semaphore(activations_arrived(s), 1);')
edit('demos/low-latency-llama/matvec_pipeline.cuh','    template <int iter_scale = 1>','    template <int iter_scale = 1, bool early_release = false>')
edit('demos/low-latency-llama/matvec_pipeline.cuh','            if ((i + 1) % iter_scale == 0) {','            if (!early_release && (i + 1) % iter_scale == 0) {')
# Shared slot release invoked by all storer lanes, after TMA completion and
# before publication. Paired outputs are released together only on odd index.
edit('demos/low-latency-llama/matvec_pipeline.cuh','    template <int iter_scale = 1, bool early_release = false>','''    template<int scale> __device__ static inline void release_output_slots(megakernel::state<Config>&s,int i) {
        for(int j=0;j<scale;j++)s.warp_arrive(outputs_finished(s,(i-j)%OUTPUT_PIPELINE_STAGES));
    }
    template <int iter_scale = 1, bool early_release = false>''')
for name,scale in [('rms_matvec_rope_append.cu',1),('upgate.cu',2)]:
 f='demos/low-latency-llama/'+name
 old='                gmem_add_release(&g.Bar['
 edit(f,old,'''
#if MK_EARLY_OUTPUT_RELEASE || MK_ASYNC_PUBLICATION
            }
            kittens::warp::sync();
#if MK_EARLY_OUTPUT_RELEASE
            pipeline::template release_output_slots<'''+str(scale)+'''>(s,output_idx);
#endif
            if (kittens::laneid() == 0) {
#endif
                megakernel::publish_output(s,&g.Bar[''')
 # Async mode also gets early release, selected by build flags.
 if scale==1:
  edit(f,'            pipeline::storer_loop(s, g);','            pipeline::template storer_loop<1,MK_EARLY_OUTPUT_RELEASE>(s, g);\n            megakernel::publication_drain(s);')
  edit(f,'            pipeline::launcher_loop(s, g);','            pipeline::launcher_loop(s, g);\n            megakernel::publication_run(s,inst.iters);')
 else:
  edit(f,'            pipeline::storer_loop<2>(s, g);','            pipeline::template storer_loop<2,MK_EARLY_OUTPUT_RELEASE>(s, g);\n            megakernel::publication_drain(s);')
  edit(f,'            pipeline::launcher_loop(s, g);','            pipeline::launcher_loop(s, g);\n            parsed_instruction inst{s};\n            megakernel::publication_run(s,inst.iters/2);')
print(w)
