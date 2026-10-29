"""Embed source 2x1 logical clusters in the VM's physical 2x2 cluster.

Only addressing changes: pair-local ranks become physical ranks at DSM/TMA/
TCGEN multicast instructions. Collective tile shapes and stages stay intact.
"""
from pathlib import Path

def prepare_cluster_overlay(cutlass: Path, output: Path):
    def emit(relative,text):
        path=output/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
    name='cute/arch/cluster_sm90.hpp';s=(cutlass/name).read_text()
    helpers=r'''
// VM logical-cluster adapter. Set collectively before entering an opcode.
CUTE_DEVICE uint32_t& faithful_pair_mode() { __shared__ uint32_t mode; return mode; }
CUTE_DEVICE uint32_t faithful_physical_cta_rank() {
  uint32_t rank; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank)); return rank;
}
CUTE_DEVICE uint32_t faithful_pair_base() {
  uint32_t mode=faithful_pair_mode();
  return mode==2 ? faithful_physical_cta_rank() : mode==1 ? (faithful_physical_cta_rank() & ~1u) : 0u;
}
CUTE_DEVICE uint32_t faithful_physical_rank(uint32_t rank) { return rank + faithful_pair_base(); }
CUTE_DEVICE uint16_t faithful_physical_mask(uint16_t mask) { return uint16_t(mask << faithful_pair_base()); }
'''
    s=s.replace('namespace cute {','namespace cute {\n'+helpers,1)
    def change_function(text,name,before,after):
        start=text.index(name+'(');end=text.index('\n}',start)+2
        fn=text[start:end]
        if fn.count(before)!=1:raise RuntimeError((name,before))
        return text[:start]+fn.replace(before,after)+text[end:]
    s=change_function(s,'block_id_in_cluster','return {x, y, z};','return faithful_pair_mode()==2 ? dim3{0,0,0} : faithful_pair_mode()==1 ? dim3{x,0,0} : dim3{x,y,z};')
    s=change_function(s,'block_rank_in_cluster','return rank;','return faithful_pair_mode()==2 ? 0u : faithful_pair_mode()==1 ? (rank & 1u) : rank;')
    s=change_function(s,'cluster_shape','return {x, y, z};','return faithful_pair_mode()==2 ? dim3{1,1,1} : faithful_pair_mode()==1 ? dim3{2,1,1} : dim3{x,y,z};')
    s=change_function(s,'set_block_rank','uint32_t result;','rank = faithful_physical_rank(rank);\n  uint32_t result;')
    emit(name,s)
    for name in ('cute/arch/copy_sm90_tma.hpp','cute/arch/copy_sm100_tma.hpp'):
        s=(cutlass/name).read_text().replace('"h"(multicast_mask)','"h"(cute::faithful_physical_mask(multicast_mask))')
        emit(name,s)
    name='cutlass/arch/barrier.h';s=(cutlass/name).read_text()
    s=s.replace('"r"(cta_id)','"r"(cute::faithful_physical_rank(cta_id))')
    s=s.replace('"h"(cta_mask)','"h"(cute::faithful_physical_mask(cta_mask))')
    s=s.replace('"r"(uint32_t(cta_mask))','"r"(uint32_t(cute::faithful_physical_mask(cta_mask)))')
    emit(name,s)
