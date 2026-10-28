"""Use hardware mbarriers for the actual storer-to-publisher queue."""
from pathlib import Path
import sys
w=Path(sys.argv[1]);p=w/'include/output_publication.cuh';s=p.read_text();start=s.index('__device__ inline int queue_acquire');s=s[:start]+'''template<class C> __device__ void publication_init(state<C>&s,int used) {
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
''';p.write_text(s)
p=w/'demos/low-latency-llama/matvec_pipeline.cuh';s=p.read_text().replace('        megakernel::publication_init(s);\n','');p.write_text(s)
p=w/'demos/low-latency-llama/rms_matvec_rope_append.cu';s=p.read_text().replace('            return pipeline::SEM_COUNT + 1;','''            megakernel::publication_init(s,pipeline::SEM_COUNT+1);
            return MK_ASYNC_PUBLICATION ? Config::DYNAMIC_SEMAPHORES : pipeline::SEM_COUNT + 1;''');p.write_text(s)
p=w/'demos/low-latency-llama/upgate.cu';s=p.read_text().replace('            return pipeline::init_semaphores(s);','''            int used=pipeline::init_semaphores(s);
            megakernel::publication_init(s,used);
            return MK_ASYNC_PUBLICATION ? Config::DYNAMIC_SEMAPHORES : used;''');p.write_text(s)
