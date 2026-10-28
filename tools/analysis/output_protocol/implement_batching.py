"""Coalesce consecutive readiness updates; flush at counter/instruction boundaries."""
from pathlib import Path
import sys
w=Path(sys.argv[1]);p=w/'include/output_publication.cuh';s=p.read_text()
s=s.replace('namespace megakernel {','''#ifndef MK_BATCHED_PUBLICATION
#define MK_BATCHED_PUBLICATION 0
#endif
namespace megakernel {''')
s=s.replace('''#if MK_ASYNC_PUBLICATION
 auto&q=publication_queue(s);q.head=0;q.tail=0;''','''#if MK_ASYNC_PUBLICATION || MK_BATCHED_PUBLICATION
 auto&q=publication_queue(s);q.head=0;q.tail=0;q.records[0]={nullptr,0,0};''')
s=s.replace('''#if MK_ASYNC_PUBLICATION
 auto&q=publication_queue(s);int tail=q.tail;''','''#if MK_BATCHED_PUBLICATION
 // The producer completes every TMA store before accumulating readiness.
 // One release RMW publishes all completed outputs for this counter.
 auto&r=publication_queue(s).records[0];
 if(r.counter!=counter && r.value){gmem_add_release(r.counter,r.value);r.value=0;}
 r.counter=counter;r.value+=value;
#elif MK_ASYNC_PUBLICATION
 auto&q=publication_queue(s);int tail=q.tail;''')
s=s.replace('''#if MK_ASYNC_PUBLICATION
 if(kittens::laneid()==0){auto&q=publication_queue(s);int tail=q.tail;''','''#if MK_BATCHED_PUBLICATION
 if(kittens::laneid()==0){auto&r=publication_queue(s).records[0];
 if(r.value)gmem_add_release(r.counter,r.value);}
 kittens::warp::sync();
#elif MK_ASYNC_PUBLICATION
 if(kittens::laneid()==0){auto&q=publication_queue(s);int tail=q.tail;''')
p.write_text(s)
