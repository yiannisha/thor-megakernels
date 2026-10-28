// Model-free VM smoke: real queue dispatch, logical IDs, page/ring reuse and NoOps.
#include "megakernel.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>

using Config = megakernel::thor_config;
struct Globals {
    megakernel::instruction_layout<Config> instructions;
    megakernel::timing_layout<Config> timings;
    int *output;
};
struct Stamp : megakernel::NoOp<Config> {
    static constexpr int opcode = 1;
    struct storer {
        __device__ static void run(const Globals &g, megakernel::state<Config> &s) {
            if (kittens::laneid() == 0)
                atomicAdd(g.output + megakernel::get_worker_id(), s.instruction()[1]);
        }
    };
};
static void check(cudaError_t status, const char *op) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", op, cudaGetErrorString(status)); std::exit(1);
    }
}
int main() {
    cudaDeviceProp props;
    check(cudaGetDeviceProperties(&props, 0), "properties");
    if (props.major != 11 || props.minor != 0) { puts("requires Thor CC11.0"); return 2; }
    auto kernel = megakernel::mk<Config, Globals, Stamp>;
    cudaFuncAttributes attr;
    check(cudaFuncGetAttributes(&attr, kernel), "kernel attributes");
    if (attr.sharedSizeBytes + Config::DYNAMIC_SHARED_MEMORY > props.sharedMemPerBlockOptin) {
        puts("shared memory payload/static/alignment exceeds optin limit"); return 3;
    }
    check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                              Config::DYNAMIC_SHARED_MEMORY), "opt-in");
    int occupancy;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, kernel, Config::NUM_THREADS, Config::DYNAMIC_SHARED_MEMORY), "occupancy");
    if (!props.cooperativeLaunch) { puts("cooperative launch unavailable"); return 6; }
    if (occupancy < 1) { puts("no resident CTA possible"); return 4; }
    int cases = 0;
    for (int workers : {1, 3, props.multiProcessorCount}) {
        if (workers > props.multiProcessorCount) continue;
        for (int length : {0, 1, 2, 3, 17, 257}) {
            for (bool only_noops : {true, false}) {
                const int allocation = workers * (length ? length : 1);
                int *instructions, *timings, *output;
                check(cudaMallocManaged(&instructions, allocation * Config::INSTRUCTION_WIDTH * sizeof(int)), "instructions");
                check(cudaMallocManaged(&timings, allocation * Config::TIMING_WIDTH * sizeof(int)), "timings");
                check(cudaMallocManaged(&output, workers * sizeof(int)), "output");
                for (int i = 0; i < allocation * Config::INSTRUCTION_WIDTH; ++i) instructions[i] = 0;
                for (int w = 0; w < workers; ++w) {
                    output[w] = 0;
                    for (int i = 0; i < length; ++i) {
                        int offset = (w * length + i) * Config::INSTRUCTION_WIDTH;
                        instructions[offset] = only_noops || i % 3 == 1 ? 0 : 1;
                        instructions[offset + 1] = w + 1; // distinguish queues, including nonsquare shapes
                    }
                }
                Globals g{kittens::make_gl<megakernel::instruction_layout<Config>>(
                              (uint64_t)instructions, 1, workers, length, Config::INSTRUCTION_WIDTH),
                          kittens::make_gl<megakernel::timing_layout<Config>>(
                              (uint64_t)timings, 1, workers, length, Config::TIMING_WIDTH), output};
                void *parameters[] = {&g};
                check(cudaLaunchCooperativeKernel(reinterpret_cast<const void *>(kernel), workers,
                      Config::NUM_THREADS, parameters, Config::DYNAMIC_SHARED_MEMORY), "cooperative VM launch");
                check(cudaGetLastError(), "VM launch");
                check(cudaDeviceSynchronize(), "VM synchronize");
                int stamps = 0;
                for (int i = 0; i < length; ++i) if (!only_noops && i % 3 != 1) ++stamps;
                for (int w = 0; w < workers; ++w) {
                    if (output[w] != stamps * (w + 1)) {
                        fprintf(stderr, "FAIL workers=%d length=%d worker=%d got=%d expected=%d\n",
                                workers, length, w, output[w], stamps * (w + 1)); return 5;
                    }
                }
                check(cudaFree(output), "free output");
                check(cudaFree(timings), "free timings");
                check(cudaFree(instructions), "free instructions");
                ++cases;
            }
        }
    }
    printf("PASS %d VM cases; physical_sms=%d resident_ctas_per_sm=%d static_smem=%zu dynamic_smem=%d registers=%d\n",
           cases, props.multiProcessorCount, occupancy, attr.sharedSizeBytes,
           Config::DYNAMIC_SHARED_MEMORY, attr.numRegs);
}
