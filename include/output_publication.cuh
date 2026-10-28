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
template<class C> __device__ void publication_init(state<C>&s,int used) {
#if MK_ASYNC_PUBLICATION
 auto&q=publication_queue(s);q.tail=0;
 for(int i=used;i<C::DYNAMIC_SEMAPHORES;i++)kittens::init_semaphore(s.semaphores()[i],1);
#endif
}
template<class C> __device__ void publish_output(state<C>&s,unsigned*counter,int value) {
#if MK_ASYNC_PUBLICATION
 auto&q=publication_queue(s);int tail=q.tail,slot=tail%8;
 kittens::wait(s.semaphores()[24+slot],((tail/8)&1)==0);
 q.records[slot]={counter,value,0};
 kittens::arrive(s.semaphores()[16+slot]);q.tail=tail+1;
#else
 gmem_add_release(counter,value);
#endif
}
template<class C> __device__ void publication_drain(state<C>&s) {
#if MK_ASYNC_PUBLICATION
 if(kittens::laneid()==0){int tail=publication_queue(s).tail;
 if(tail>0)kittens::wait(s.semaphores()[24+(tail-1)%8],((tail-1)/8)&1);}
 kittens::warp::sync();
#endif
}
template<class C> __device__ void publication_run(state<C>&s,int count) {
#if MK_ASYNC_PUBLICATION
 if(kittens::laneid()==0){auto&q=publication_queue(s);
 for(int head=0;head<count;head++){
 int slot=head%8;
 kittens::wait(s.semaphores()[16+slot],(head/8)&1);
 auto record=q.records[slot];
 gmem_add_release(record.counter,record.value);
 kittens::arrive(s.semaphores()[24+slot]);
 }}
#endif
}
}
