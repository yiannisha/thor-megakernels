#pragma once
#include "cute/tensor.hpp"
#include "kernel/fmha_tile_scheduler.hpp"
namespace faithful {
// One source IndividualTileScheduler work unit per VM wave/CTA. This keeps
// the source nonpersistent mainloop/epilogue shared-memory union unchanged.
struct FmhaScheduler:cutlass::fmha::kernel::IndividualTileScheduler {
    using Base=cutlass::fmha::kernel::IndividualTileScheduler;
    Params p;int tile;
    CUTLASS_DEVICE FmhaScheduler(Params const& params,int index):Base(params),p(params),tile(index){valid_=index<int(p.grid.x*p.grid.y*p.grid.z);}
    CUTLASS_DEVICE auto get_block_coord(){
        int m=tile%p.grid.x;int h=(tile/p.grid.x)%p.grid.y;int b=tile/(p.grid.x*p.grid.y);
        return cute::make_coord(m,cute::_0{},cute::make_coord(h,b));
    }
};
}
