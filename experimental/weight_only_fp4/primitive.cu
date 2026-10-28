// W4A16 feasibility: packed NVFP4 remains in global memory; only the current
// tile expands in shared memory. BF16 activations are never quantized to FP4.
#include "../../tools/analysis/nvfp4/primitive.cu"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cute/arch/copy_sm100.hpp"
namespace weight_only {
using BF=cutlass::bfloat16_t;
using Mma=decltype(cute::make_tiled_mma(cute::SM100_MMA_F16BF16_SS<BF,BF,float,128,8,cute::UMMA::Major::K,cute::UMMA::Major::K>{}));
using AShape=decltype(cute::partition_shape_A(Mma{},cute::Shape<cute::_128,cute::_128>{}));
using BShape=decltype(cute::partition_shape_B(Mma{},cute::Shape<cute::_8,cute::_128>{}));
using ALayout=decltype(cute::UMMA::tile_to_mma_shape(cute::UMMA::Layout_K_SW128_Atom<BF>{},AShape{}));
using BLayout=decltype(cute::UMMA::tile_to_mma_shape(cute::UMMA::Layout_K_SW128_Atom<BF>{},BShape{}));
struct Shared {
 alignas(128) cute::ArrayEngine<BF,cute::cosize_v<ALayout>> a;
 alignas(128) cute::ArrayEngine<BF,cute::cosize_v<BLayout>> b;
 alignas(16) uint64_t barrier;
 uint32_t tmem;
};
constexpr int allocated_shared=128*1024;
static_assert(sizeof(Shared)<=allocated_shared);
__global__ __launch_bounds__(256,1) void linear(const unsigned char* weights,const unsigned char* scales,
 const __nv_bfloat16* input,__nv_bfloat16* output,int m,int k,float weight_scale){
 extern __shared__ __align__(128) char bytes[];auto& sm=*reinterpret_cast<Shared*>(bytes);
 int tid=threadIdx.x,lane=tid%32,warp=tid/32;
 auto mma=Mma{};auto cta=mma.get_slice(0);
 auto sa=cute::make_tensor(cute::make_smem_ptr(sm.a.begin()),ALayout{});
 auto sb=cute::make_tensor(cute::make_smem_ptr(sm.b.begin()),BLayout{});
 auto ra=cta.make_fragment_A(sa);auto rb=cta.make_fragment_B(sb);
 auto gc=cute::make_tensor(cute::make_gmem_ptr((float*)nullptr),cute::make_layout(cute::Shape<cute::_128,cute::_8>{},cute::Stride<cute::_1,cute::_128>{}));
 auto pc=cta.partition_C(gc);auto acc=cta.make_fragment_C(pc);
 if(warp==0)cute::TMEM::Allocator1Sm{}.allocate(512,&sm.tmem);
 if(tid==0)cute::initialize_barrier(sm.barrier,1);
 __syncthreads();asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
 acc.data()=sm.tmem;int phase=0;
 for(int tile=blockIdx.x;tile<(m+127)/128;tile+=gridDim.x){
  mma.accumulate_=cute::UMMA::ScaleOut::Zero;
  for(int start=0;start<k;start+=128){
   for(int pair=tid;pair<128*128/2;pair+=blockDim.x){
    int row=pair/64,col=(pair%64)*2,global_row=tile*128+row;
    float v0=0,v1=0;
    if(global_row<m){
     unsigned char packed=weights[(global_row*k+start+col)/2];
     __half2 halfs=__half2(__nv_cvt_fp4x2_to_halfraw2(packed,__NV_E2M1));
     __nv_fp8_e4m3 scale;scale.__x=scales[global_row*(k/16)+(start+col)/16];
     v0=__low2float(halfs)*float(scale);v1=__high2float(halfs)*float(scale);
    }
    sa(cute::make_coord(row,col%16),0,col/16)=BF(v0);
    sa(cute::make_coord(row,(col+1)%16),0,(col+1)/16)=BF(v1);
   }
   for(int i=tid;i<8*128;i+=blockDim.x){int row=i/128,col=i%128;
    sb(cute::make_coord(row,col%16),0,col/16)=BF(row==0?__bfloat162float(input[start+col]):0.f);
   }
   asm volatile("fence.proxy.async.shared::cta;" ::: "memory");__syncthreads();
   if(warp==0){
    #pragma unroll
    for(int b=0;b<cute::size<2>(ra);++b){cute::gemm(mma,ra(cute::_,cute::_,b),rb(cute::_,cute::_,b),acc);mma.accumulate_=cute::UMMA::ScaleOut::One;}
    cutlass::arch::umma_arrive(&sm.barrier);
   }
   cute::wait_barrier(sm.barrier,phase);phase^=1;
  }
  if(warp<4){uint32_t value;cute::SM100_TMEM_LOAD_32dp32b1x::copy(sm.tmem+((warp*32)<<16),value);
   asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
   int row=tile*128+warp*32+lane;if(row<m)output[row]=__float2bfloat16(__uint_as_float(value)*weight_scale);
  }
  __syncthreads();
 }
 if(warp==0){cute::TMEM::Allocator1Sm{}.release_allocation_lock();cute::TMEM::Allocator1Sm{}.free(sm.tmem,512);}
}
}
extern "C" int wo_initialize(){
 auto e=cudaFuncSetAttribute(weight_only::linear,cudaFuncAttributeMaxDynamicSharedMemorySize,weight_only::allocated_shared);
 if(e!=cudaSuccess)return e;
 int occupancy;e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,weight_only::linear,256,weight_only::allocated_shared);
 if(e!=cudaSuccess)return e;return occupancy==1?0:int(cudaErrorInvalidConfiguration);
}
extern "C" int wo_run(void* weights,void* scales,void* input,void* output,int m,int k,float weight_scale,int workers,void* stream){
 if(m<=0 || k<=0 || k%128 || workers<=0)return cudaErrorInvalidValue;
 weight_only::linear<<<workers,256,weight_only::allocated_shared,(cudaStream_t)stream>>>((unsigned char*)weights,(unsigned char*)scales,(__nv_bfloat16*)input,(__nv_bfloat16*)output,m,k,weight_scale);
 return cudaGetLastError();
}
