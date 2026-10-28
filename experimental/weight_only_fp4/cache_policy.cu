#include <cuda_runtime.h>
#include <algorithm>
extern "C" int wo_cache_limits(int* limits){
 int device;auto e=cudaGetDevice(&device);if(e!=cudaSuccess)return e;
 cudaDeviceProp p;e=cudaGetDeviceProperties(&p,device);if(e!=cudaSuccess)return e;
 limits[0]=p.l2CacheSize;limits[1]=p.persistingL2CacheMaxSize;limits[2]=p.accessPolicyMaxWindowSize;return 0;
}
extern "C" int wo_cache_policy(void* stream,void* cache,size_t bytes,size_t* previous){
 auto e=cudaDeviceGetLimit(previous,cudaLimitPersistingL2CacheSize);if(e!=cudaSuccess)return e;
 int limits[3];e=(cudaError_t)wo_cache_limits(limits);if(e!=cudaSuccess)return e;
 size_t size=std::min(bytes,size_t(limits[1]));if(!size)return cudaErrorNotSupported;
 e=cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,size);if(e!=cudaSuccess)return e;
 cudaStreamAttrValue attr{};attr.accessPolicyWindow.base_ptr=cache;
 attr.accessPolicyWindow.num_bytes=std::min(bytes,size_t(limits[2]));
 attr.accessPolicyWindow.hitRatio=std::min(1.f,float(size)/attr.accessPolicyWindow.num_bytes);
 attr.accessPolicyWindow.hitProp=cudaAccessPropertyPersisting;attr.accessPolicyWindow.missProp=cudaAccessPropertyStreaming;
 return cudaStreamSetAttribute((cudaStream_t)stream,cudaStreamAttributeAccessPolicyWindow,&attr);
}
extern "C" int wo_cache_reset(void* stream,size_t previous){
 cudaStreamAttrValue attr{};auto e=cudaStreamSetAttribute((cudaStream_t)stream,cudaStreamAttributeAccessPolicyWindow,&attr);if(e!=cudaSuccess)return e;
 e=cudaCtxResetPersistingL2Cache();if(e!=cudaSuccess)return e;
 return cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,previous);
}
#include <vector>
extern "C" int wo_graph_cache_policy(void* graph,void* cache,size_t bytes){
 int limits[3];auto e=(cudaError_t)wo_cache_limits(limits);if(e!=cudaSuccess)return e;
 cudaKernelNodeAttrValue attr{};attr.accessPolicyWindow.base_ptr=cache;
 attr.accessPolicyWindow.num_bytes=std::min(bytes,size_t(limits[2]));
 attr.accessPolicyWindow.hitRatio=std::min(1.f,float(limits[1])/attr.accessPolicyWindow.num_bytes);
 attr.accessPolicyWindow.hitProp=cudaAccessPropertyPersisting;attr.accessPolicyWindow.missProp=cudaAccessPropertyStreaming;
 size_t count=0;e=cudaGraphGetNodes((cudaGraph_t)graph,nullptr,&count);if(e!=cudaSuccess)return e;
 std::vector<cudaGraphNode_t> nodes(count);e=cudaGraphGetNodes((cudaGraph_t)graph,nodes.data(),&count);if(e!=cudaSuccess)return e;
 for(auto node:nodes){cudaGraphNodeType type;e=cudaGraphNodeGetType(node,&type);if(e!=cudaSuccess)return e;
  if(type==cudaGraphNodeTypeKernel){e=cudaGraphKernelNodeSetAttribute(node,cudaKernelNodeAttributeAccessPolicyWindow,&attr);if(e!=cudaSuccess)return e;}}
 return 0;
}
