#pragma once
#include "../gmem_barrier.cuh"
#include "../util.cuh"
#include "page_allocator.cuh"
#include "semaphore_constructor.cuh"

namespace megakernel::ready {
// Immutable CSR: offsets[N+1], children[E], initial_indegrees[N].
// Mutable: head, tail, completed, slots[N], pending[N], visits[N].
template <typename Globals> __device__ inline int tasks(const Globals &g) { return g.instructions.rows(); }
template <typename Globals> __device__ inline int edges(const Globals &g) { return g.ready_metadata.cols() - 2*tasks(g) - 1; }
template <typename Globals> __device__ inline int *slots(const Globals &g) { return g.ready_state.raw_ptr + 3; }
template <typename Globals> __device__ inline int *pending(const Globals &g) { return slots(g) + tasks(g); }
template <typename Globals> __device__ inline int *visits(const Globals &g) { return pending(g) + tasks(g); }
__device__ inline void fail() { asm volatile("trap;" ::: "memory"); }
__device__ inline auto atomic(int *p) { return cuda::atomic_ref<int, cuda::thread_scope_device>(*p); }

template <typename Globals> __device__ inline void initialize(const Globals &g) {
    int n=tasks(g), e=edges(g), stride=gridDim.x*blockDim.x;
    int thread=blockIdx.x*blockDim.x+threadIdx.x;
    for (int i=thread;i<3+3*n;i+=stride) {
        int value=0;
        if (i>=3 && i<3+n) value=-1;
        else if (i>=3+n && i<3+2*n) value=g.ready_metadata.raw_ptr[n+1+e+i-3-n];
        g.ready_state.raw_ptr[i]=value;
    }
}
template <typename Globals> __device__ inline void push(const Globals &g,int task) {
    int slot=atomic(g.ready_state.raw_ptr+1).fetch_add(1,cuda::memory_order_acq_rel);
    if (slot<0 || slot>=tasks(g)) fail();
    atomic(slots(g)+slot).store(task,cuda::memory_order_release);
}
template <typename Globals> __device__ inline void publish_roots(const Globals &g) {
    for (int i=blockIdx.x*blockDim.x+threadIdx.x;i<tasks(g);i+=gridDim.x*blockDim.x)
        if (pending(g)[i]==0) push(g,i);
}
template <typename Config,typename Globals> __device__ inline int pop(const Globals &g) {
    int task=-1;
    gmem_wait<Config>([&] {
        int head=atomic(g.ready_state.raw_ptr).load(cuda::memory_order_acquire);
        int tail=atomic(g.ready_state.raw_ptr+1).load(cuda::memory_order_acquire);
        if (head<tail) {
            int candidate=atomic(slots(g)+head).load(cuda::memory_order_acquire);
            if (candidate>=0 && atomic(g.ready_state.raw_ptr).compare_exchange_weak(
                    head,head+1,cuda::memory_order_acq_rel,cuda::memory_order_acquire)) {
                if (candidate>=tasks(g) || atomic(pending(g)+candidate).load(cuda::memory_order_acquire)!=0) fail();
                if (atomic(visits(g)+candidate).fetch_add(1,cuda::memory_order_acq_rel)!=0) fail();
                task=candidate;
                return false;
            }
        }
        return atomic(g.ready_state.raw_ptr+2).load(cuda::memory_order_acquire)!=tasks(g);
    });
    return task; // -1 only when every task has completed.
}
template <typename Globals> __device__ inline void complete(const Globals &g,int task) {
    int n=tasks(g), e=edges(g);
    int begin=g.ready_metadata.raw_ptr[task], end=g.ready_metadata.raw_ptr[task+1];
    if (begin<0 || end<begin || end>e) fail();
    for (int i=begin;i<end;i++) {
        int child=g.ready_metadata.raw_ptr[n+1+i];
        if (child<0 || child>=n) fail();
        int before=atomic(pending(g)+child).fetch_sub(1,cuda::memory_order_acq_rel);
        if (before<=0) fail();
        if (before==1) push(g,child);
    }
    int before=atomic(g.ready_state.raw_ptr+2).fetch_add(1,cuda::memory_order_acq_rel);
    if (before>=n) fail();
}
} // namespace megakernel::ready

namespace megakernel::controller {
template <typename Config,typename Globals,typename... Ops>
__device__ inline void ready_loop(const Globals &g,state<Config> &s) {
    static_assert(Config::COOPERATIVE_LAUNCH && Config::SERIAL_INSTRUCTION_HANDOFF);
    static_assert(!Config::TIMING_RECORD_ENABLED, "ready dispatch timing schema is not implemented");
    const int lane=kittens::laneid();
    for (s.instruction_index=0,s.instruction_ring=0;s.instruction_index<=ready::tasks(g);
         ++s.instruction_index,s.instruction_ring=ring_advance<Config::INSTRUCTION_PIPELINE_STAGES>(s.instruction_ring)) {
        int task=-1;
        if (lane==0) task=ready::pop<Config>(g);
        task=__shfl_sync(0xffffffff,task,0);
        if (task>=0) {
            // The table is shared, unlike the static per-worker queues.
            s.instruction()[lane]=g.instructions[kittens::coord<>{0,task,lane}];
            if (lane<Config::NUM_PAGES) {
                if (s.instruction_index==0) s.pid_order()[lane]=lane;
                else {
                    int previous=ring_retreat<Config::INSTRUCTION_PIPELINE_STAGES>(s.instruction_ring);
                    int query=lane;
                    int lid=dispatch_op<page_allocator_op_dispatcher<Config,Globals>::dispatcher,Ops...>::template run<
                        int,Config,Globals,typename Config::instruction_t,int>(
                            s.all_instructions[previous].instructions[0],g,s.all_instructions[previous].instructions,query);
                    s.pid_order()[lane]=s.all_instructions[previous].pid_order[lid];
                }
            }
        } else s.instruction()[lane]=-1;
        __syncwarp();
        int count=0;
        if (task>=0 && s.instruction()[0]!=0 && lane==0)
            count=dispatch_op<semaphore_constructor_op_dispatcher<Config,Globals>::dispatcher,Ops...>::template run<
                int,Config,Globals,state<Config>>(s.instruction()[0],g,s);
        count=__shfl_sync(0xffffffff,count,0);
        __syncwarp();
        if (lane<Config::INSTRUCTION_BARRIER_PARTIES_PER_WARP) kittens::arrive(s.instruction_arrived[s.instruction_ring]);
        kittens::wait(s.instruction_finished[s.instruction_ring],
                      (s.instruction_index/Config::INSTRUCTION_PIPELINE_STAGES)&1);
        if (lane<count) invalidate_semaphore(s.semaphores()[lane]);
        __syncwarp();
        if (task<0) return;
        // Every execution thread released instruction_finished after completing
        // its stores/TMA operations. This lane acquired that publication before
        // the device-scope RMW chain makes children ready.
        if (lane==0) ready::complete(g,task);
        __syncwarp();
    }
    ready::fail(); // No worker can execute more than N distinct tasks plus stop.
}
} // namespace megakernel::controller
