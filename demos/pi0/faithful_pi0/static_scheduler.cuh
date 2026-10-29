#pragma once
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/kernel/sm100_tile_scheduler.hpp"

namespace faithful {
struct StaticSchedulerTag {};
// Replace host-launched CLC work discovery with the resident VM cluster stream.
// The GEMM's mainloop, epilogue, pipelines and two-CTA MMA remain source code.
template<class Cluster, uint32_t Stages>
struct StaticScheduler : cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100<Cluster,Stages> {
    using Base = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100<Cluster,Stages>;
    static constexpr bool IsDynamicPersistent = false;
    struct Params : Base::Params { int clusters_m=0,clusters_n=0,batches=1; };
    struct WorkTileInfo : Base::WorkTileInfo { int ordinal=0; };
    using Arguments=typename Base::Arguments;
    Params p; dim3 rank;
    CUTLASS_DEVICE StaticScheduler(typename Base::CLCResponse* ptr,Params const& params,dim3 r)
        :Base(ptr,params,r),p(params),rank(r){}
    template<class Problem,class Tile,class Atom,class CS>
    static Params to_underlying_arguments(Problem problem,Tile tile,Atom atom,CS cs,
        cutlass::KernelHardwareInfo const& hw,Arguments const& args,void* workspace=nullptr) {
        Params p;
        static_cast<typename Base::Params&>(p)=Base::to_underlying_arguments(problem,tile,atom,cs,hw,args,workspace);
        p.clusters_m=cute::ceil_div(cute::get<0>(problem),cute::get<0>(tile)/cute::get<0>(atom)*cute::size<0>(cs));
        p.clusters_n=cute::ceil_div(cute::get<1>(problem),cute::get<1>(tile)/cute::get<1>(atom)*cute::size<1>(cs));
        p.batches=cute::get<3>(problem);return p;
    }
    CUTLASS_DEVICE WorkTileInfo work(int i) const {
        WorkTileInfo w;
        w.ordinal=i;w.is_valid_tile=i<p.clusters_m*p.clusters_n*p.batches;
        w.M_idx=(i%p.clusters_m)*int(cute::size<0>(Cluster{}))+rank.x;
        w.N_idx=((i/p.clusters_m)%p.clusters_n)*int(cute::size<1>(Cluster{}))+rank.y;
        w.L_idx=i/(p.clusters_m*p.clusters_n);return w;
    }
    template<class CS> CUTLASS_DEVICE WorkTileInfo initial_work_tile_info(CS) {
        int cx=gridDim.x/int(cute::size<0>(Cluster{}));
        return work(blockIdx.x/int(cute::size<0>(Cluster{}))+cx*(blockIdx.y/int(cute::size<1>(Cluster{}))));
    }
    template<class Pipeline,class State> CUTLASS_DEVICE auto fetch_next_work(WorkTileInfo w,Pipeline&,State) {
        int clusters=gridDim.x*gridDim.y*gridDim.z/int(cute::size(Cluster{}));
        return cute::make_tuple(work(w.ordinal+clusters),false);
    }
};
}
namespace cutlass::gemm::kernel::detail {
template<class Arch,class Tile,class Cluster,uint32_t Stages>
struct TileSchedulerSelector<faithful::StaticSchedulerTag,Arch,Tile,Cluster,Stages> {
    using Scheduler=faithful::StaticScheduler<Cluster,Stages>;
};
}
