#include "format_storage.cuh"
#include <cstdio>
using namespace megakernel::quantized;
using BF = weight_stage<bf16_format,16,2048>;
using FP = weight_stage<nvfp4_format,16,2048>;
static_assert(BF::bytes == 65536);
static_assert(FP::bytes == 18432);
using P = page_plan<FP,4096,3,8192>;
static_assert(P::stage_pages == 3 && P::pages == 10);
static_assert(P::stage_offset(2) + P::stage_pages*8192 == P::allocated_bytes);
using RmsPlan = page_plan<FP,8192,3,2048>;
static_assert(RmsPlan::pages == 31);
using LargeFP = weight_stage<nvfp4_format,16,8192>;
using LargePlan = page_plan<LargeFP,32768,3,8192>;
static_assert(LargePlan::pages == 31);
int main(){
 std::printf("{\"bf16_stage_bytes\":%zu,\"nvfp4_stage_bytes\":%zu,\"nvfp4_page8192_allocation\":%d}\n",BF::bytes,FP::bytes,P::allocated_bytes);
}
