// Generic batch-one linear feasibility probe. CUTLASS implements tcgen05/TMEM.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cstdio>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
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
#ifndef NV_STAGES
#define NV_STAGES 0
#endif
#if NV_STAGES > 0
using NvStagePolicy = cutlass::gemm::collective::StageCount<NV_STAGES>;
#else
using NvStagePolicy = cutlass::gemm::collective::StageCountAutoCarveout<int(sizeof(Epilogue::SharedStorage))>;
#endif
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
 cutlass::arch::Sm100,cutlass::arch::OpClassBlockScaledTensorOp,
 NvElement,cutlass::layout::RowMajor,32,NvElement,cutlass::layout::ColumnMajor,32,float,NvTile,NvCluster,
 NvStagePolicy,
 cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
#ifdef NV_STATIC_RUNTIME
using SchedulerTag = cutlass::gemm::StaticPersistentScheduler;
#else
using SchedulerTag = void;
#endif
using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Mainloop,Epilogue,SchedulerTag>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
using SFConfig = Mainloop::Sm1xxBlkScaledConfig;
using SFLayout = Mainloop::LayoutSFA;

// One warp per 16-value group; no global BF16 expansion. Tensor scale is a
// fixed preselected positive scalar, separate from per-block E4M3 scales.
template<class Layout>
__global__ void quantize(const __nv_bfloat16* input, unsigned char* packed,
 unsigned char* canonical, unsigned char* native, float* decoded,
 int rows,int k,float tensor_scale,Layout layout) {
 int group=(blockIdx.x*blockDim.x+threadIdx.x)/32;
 int lane=threadIdx.x%32;
 int groups=rows*(k/16);
 if(group>=groups) return;
 int row=group/(k/16), col=(group%(k/16))*16;
 float v=lane<16?__bfloat162float(input[row*k+col+lane])/tensor_scale:0.f;
 float a=fabsf(v);
 for(int d=16;d;d>>=1) a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,d));
 // Round scale first, then quantize against that exact scale.
 __nv_fp8_e4m3 sf(fmaxf(a/6.f,1.f/512.f));
 float scale=float(sf);
 unsigned char q=__nv_cvt_float_to_fp4(v/scale,__NV_E2M1,cudaRoundNearest);
 if(lane==0){
  canonical[group]=sf.__x;
  native[layout(make_coord(row,col,0))]=sf.__x;
 }
 unsigned char next=__shfl_down_sync(0xffffffff,unsigned(q),1);
 if(lane<16 && lane%2==0) packed[(row*k+col+lane)/2]=q|(next<<4);
 if(lane<16 && decoded) {
  cutlass::float_e2m1_t fp; fp.raw()=q;
  decoded[row*k+col+lane]=float(fp)*scale*tensor_scale;
 }
}
__device__ float unpack(unsigned char b) {
 __half_raw h=__nv_cvt_fp4_to_halfraw(b&15,__NV_E2M1); return __half2float(__half(h));
}
template<bool Packed>
__global__ void simt(const __nv_bfloat16* w,const unsigned char* p,
 const unsigned char* scales,const __nv_bfloat16* x,__nv_bfloat16* y,
 int m,int k,float ts) {
 int row=blockIdx.x*8+threadIdx.x/32;
 int lane=threadIdx.x%32;
 float acc=0;
 if(row<m) for(int c=lane;c<k;c+=32) {
  float weight;
  if constexpr(Packed){
   __nv_fp8_e4m3 s; s.__x=scales[row*(k/16)+c/16];
   weight=unpack(p[(row*k+c)/2]>>((c%2)*4))*float(s)*ts;
  } else weight=__bfloat162float(w[row*k+c]);
  acc=fmaf(weight,__bfloat162float(x[c]),acc);
 }
 for(int d=16;d;d>>=1) acc+=__shfl_down_sync(0xffffffff,acc,d);
 if(lane==0 && row<m) y[row]=__float2bfloat16(acc);
}
struct Plan { Gemm gemm; void* workspace=nullptr; int m,k; };
static thread_local char error_text[512];
extern "C" const char* nv_error(){return error_text;}
extern "C" long nv_sf_bytes(int rows,int k) {
 auto l=SFConfig::tile_atom_to_shape_SFA(make_shape(rows,1,k,1));
 return size(filter_zeros(l));
}
extern "C" int nv_quantize(void* input,void* packed,void* canonical,void* native,
 void* decoded,int rows,int k,float ts,void* stream) {
 auto l=SFConfig::tile_atom_to_shape_SFA(make_shape(rows,1,k,1));
 quantize<<<(rows*(k/16)+7)/8,256,0,(cudaStream_t)stream>>>(
 (__nv_bfloat16*)input,(unsigned char*)packed,(unsigned char*)canonical,
 (unsigned char*)native,(float*)decoded,rows,k,ts,l);
 return cudaGetLastError();
}
extern "C" int nv_simt(void* w,void* p,void* scales,void* x,void* y,
 int m,int k,float ts,int packed,void* stream) {
 if(packed) simt<true><<<(m+7)/8,256,0,(cudaStream_t)stream>>>(
 (__nv_bfloat16*)w,(unsigned char*)p,(unsigned char*)scales,(__nv_bfloat16*)x,(__nv_bfloat16*)y,m,k,ts);
 else simt<false><<<(m+7)/8,256,0,(cudaStream_t)stream>>>(
 (__nv_bfloat16*)w,nullptr,nullptr,(__nv_bfloat16*)x,(__nv_bfloat16*)y,m,k,ts);
 return cudaGetLastError();
}
extern "C" void* nv_prepare(void* w,void* x,void* sw,void* sx,void* y,
 int m,int k,float alpha,void* stream) {
 auto* p=new Plan; p->m=m;p->k=k;
 auto sa=cutlass::make_cute_packed_stride(Kernel::StrideA{}, {m,k,1});
 auto sb=cutlass::make_cute_packed_stride(Kernel::StrideB{}, {1,k,1});
 auto sd=cutlass::make_cute_packed_stride(Kernel::StrideD{}, {m,1,1});
 auto la=SFConfig::tile_atom_to_shape_SFA(make_shape(m,1,k,1));
 auto lb=SFConfig::tile_atom_to_shape_SFB(make_shape(m,1,k,1));
 Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,{m,1,k,1},
  {(NvElement::DataType*)w,sa,(NvElement::DataType*)x,sb,(NvElement::ScaleFactorType*)sw,la,(NvElement::ScaleFactorType*)sx,lb},
  {{alpha,0.f},(cutlass::bfloat16_t*)y,sd,(cutlass::bfloat16_t*)y,sd}};
#ifdef NV_STATIC_RUNTIME
 int device; cudaGetDevice(&device); cudaDeviceProp properties;
 cudaGetDeviceProperties(&properties,device);
 args.hw_info.device_id=device;
 args.hw_info.sm_count=properties.multiProcessorCount;
#endif
 auto status=p->gemm.can_implement(args);
 if(status==cutlass::Status::kSuccess){
  auto bytes=Gemm::get_workspace_size(args);
  auto err=cudaMalloc(&p->workspace,bytes?bytes:1);
  if(err!=cudaSuccess){snprintf(error_text,512,"allocation: %s",cudaGetErrorString(err));delete p;return nullptr;}
  status=p->gemm.initialize(args,p->workspace,(cudaStream_t)stream);
 }
 if(status!=cutlass::Status::kSuccess){snprintf(error_text,512,"CUTLASS: %s",cutlass::cutlassGetStatusString(status));cudaFree(p->workspace);delete p;return nullptr;}
 return p;
}
extern "C" int nv_run(void* handle,void* stream){return int(((Plan*)handle)->gemm.run((cudaStream_t)stream));}
extern "C" void nv_destroy(void* handle){auto* p=(Plan*)handle;if(p){cudaFree(p->workspace);delete p;}}

extern "C" int nv_stages(){return Mainloop::DispatchPolicy::Stages;}
