#pragma once
// Native NVFP4 instruction layouts; no standalone scheduler executes in the VM.
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
namespace megakernel::nvfp4 {
using namespace cute;
using NvElement = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using NvTile = Shape<_128,_64,_256>;
using NvCluster = Shape<_1,_1,_1>;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
 cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp, NvTile, NvCluster,
 cutlass::epilogue::collective::EpilogueTileAuto,float,float,
 cutlass::bfloat16_t,cutlass::layout::ColumnMajor,8,
 cutlass::bfloat16_t,cutlass::layout::ColumnMajor,8,
 cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using NvStagePolicy = cutlass::gemm::collective::StageCount<2>;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
 cutlass::arch::Sm100,cutlass::arch::OpClassBlockScaledTensorOp,
 NvElement,cutlass::layout::RowMajor,32,NvElement,cutlass::layout::ColumnMajor,32,float,NvTile,NvCluster,
 NvStagePolicy,
 cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using SFConfig = Mainloop::Sm1xxBlkScaledConfig;
using SFLayout = Mainloop::LayoutSFA;

namespace mixed8 {
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
 cutlass::arch::Sm100,cutlass::arch::OpClassBlockScaledTensorOp,
 cute::tuple<cutlass::float_e2m1_t,cutlass::float_ue8m0_t>,cutlass::layout::RowMajor,32,
 cute::tuple<cutlass::float_e4m3_t,cutlass::float_ue8m0_t>,cutlass::layout::ColumnMajor,16,
 float,NvTile,NvCluster,NvStagePolicy,cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using SFConfig=Mainloop::Sm1xxBlkScaledConfig;
}

}
