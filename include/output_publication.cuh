#pragma once
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
