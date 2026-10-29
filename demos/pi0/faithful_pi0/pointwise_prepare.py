"""Lift source kernels to device functions with VM-supplied work coordinates."""

def kernel(text,name):
    start=text.index('__global__ void '+name)
    brace=text.index('{',start);depth=1;i=brace+1
    while depth:
        depth+=(text[i]=='{')-(text[i]=='}');i+=1
    return text[start:i]

def prepare_pointwise(flash,out):
    norm=(flash/'csrc/kernels/norm.cu').read_text()
    # Include the template declaration immediately before each norm body.
    functions=[]
    for name in ('rms_norm_kernel','layer_norm_kernel'):
        body='template<typename T>\n'+kernel(norm,name)
        body=body.replace('__global__','__device__').replace('float eps)', 'float eps, int row, float* shared)')
        body=body.replace('    int row = blockIdx.x;','').replace('    extern __shared__ float shared[];','')
        functions.append(body)
    softmax=(out/'softmax_bf16.cu').read_text()
    body=kernel(softmax,'softmax_state_masked_bf16_kernel').replace('__global__','__device__')
    body=body.replace('int pad_start)', 'int pad_start, int row)').replace('    int row = blockIdx.x;','')
    functions.append(body)
    for filename,name in (('patch_bf16.cu','faithful_patch_im2col_kernel'),('rope_bf16.cu','qkv_split_rope_kvcache_bf16_kernel'),('patch_bias_bf16.cu','faithful_patch_bias_pos_impl_kernel')):
        body=kernel((out/filename).read_text(),name).replace('__global__','__device__')
        brace=body.index('{');head=body[:brace].rstrip();head=head[:head.rindex(')')]+', int idx)\n'
        body=head+body[brace:]
        body=body.replace('    int idx = blockIdx.x * blockDim.x + threadIdx.x;','')
        functions.append(body)
    text='#pragma once\n#include "common.cuh"\nnamespace faithful_pointwise {\n#define SM_WARP_SIZE 32\n#define SM_MAX_COLS 1024\n#define SM_ITERS 32\n'+'\n'.join(functions)+'\n#undef SM_WARP_SIZE\n#undef SM_MAX_COLS\n#undef SM_ITERS\n}\n'
    (out/'native_pointwise_source.cuh').write_text(text)
