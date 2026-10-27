#pragma once

#include "kittens.cuh"

namespace megakernel {

struct default_config {
    // Instruction pipeline
    static constexpr int INSTRUCTION_PIPELINE_STAGES = 2;
    static constexpr int INSTRUCTION_BARRIER_PARTIES_PER_WARP = 1;

    // num bits required to represent num pipeline stages
    static constexpr int INSTRUCTION_PIPELINE_STAGES_BITS = 1;

    static constexpr int INSTRUCTION_WIDTH = 32; // 128 bytes per instruction.
    using instruction_t = int[INSTRUCTION_WIDTH];

    // Timing info
    static constexpr int TIMING_WIDTH = 128;
    using timing_t = int[TIMING_WIDTH];

    // How many semaphores are available for dynamic use?
    static constexpr int DYNAMIC_SEMAPHORES = 32;

    // One controller warp, one load warp, one store warp, and one mma warp.
    static constexpr int NUM_CONSUMER_WARPS = 16;
    static constexpr int NUM_WARPS = 4 + NUM_CONSUMER_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * ::kittens::WARP_THREADS;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int CLUSTER_BLOCKS = 1;
    static constexpr int MAX_SHARED_MEMORY = ::kittens::MAX_SHARED_MEMORY;

    // Shared memory declared statically
    static constexpr int SCRATCH_BYTES = 4096;
    static constexpr int STATIC_SHARED_MEMORY =
        512 + INSTRUCTION_PIPELINE_STAGES *
                  (SCRATCH_BYTES + (INSTRUCTION_WIDTH + TIMING_WIDTH) * 4 +
                   DYNAMIC_SEMAPHORES * 8);
    static constexpr int DYNAMIC_SHARED_MEMORY =
        ::kittens::MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    // Shared memory declared dynamically
    static constexpr int WEIGHT_PIPELINE_STAGES = 3;
    static constexpr int PAGE_SIZE = 16384;
    static constexpr int NUM_PAGES = DYNAMIC_SHARED_MEMORY / PAGE_SIZE;
    static_assert(NUM_PAGES == 13, "NUM_PAGES must be 13");

    static constexpr bool TIMING_RECORD_ENABLED = false;

    static constexpr int GMEM_SPIN_LOOP_SLEEP_NANOS = 20;

    static constexpr bool MATCH_TORCH_BF16_RMS = false;
    static constexpr bool ORDERED_DOWNPROJ = false;
    static constexpr bool COOPERATIVE_LAUNCH = false;
    static constexpr bool FP32_DOWNPROJ = false;
    static constexpr bool MATCH_TORCH_BF16_EPILOGUES = false;
    static constexpr bool COMPENSATED_ATTENTION = false;
    static constexpr bool FP64_RMS = false;
    static constexpr bool STATIC_REGISTER_ALLOCATION = false;
    static constexpr int CONSUMER_REGISTERS = 104;
    static constexpr int NON_CONSUMER_REGISTERS = 64;
};
#ifdef MK_THOR
#ifndef MK_WEIGHT_PIPELINE_STAGES
#define MK_WEIGHT_PIPELINE_STAGES 3
#endif
#ifndef MK_PAGE_COUNT
#define MK_PAGE_COUNT (1 + 4 * MK_WEIGHT_PIPELINE_STAGES)
#endif
#ifndef MK_ACTIVE_WORKERS
#define MK_ACTIVE_WORKERS 20
#endif
#ifndef MK_CONSUMER_WARPS
#define MK_CONSUMER_WARPS 16
#endif
#ifndef MK_CONSUMER_REGISTERS
#define MK_CONSUMER_REGISTERS 104
#endif
#ifndef MK_SERVICE_REGISTERS
#define MK_SERVICE_REGISTERS 64
#endif
#ifndef MK_BACKOFF_NS
#define MK_BACKOFF_NS 20
#endif
#ifndef MK_RECORD_TIMING
#define MK_RECORD_TIMING 0
#endif
#ifndef MK_INSTRUCTION_BARRIER_PARTIES
#define MK_INSTRUCTION_BARRIER_PARTIES 32
#endif
#ifndef MK_MATCH_TORCH_BF16_RMS
#define MK_MATCH_TORCH_BF16_RMS 1
#endif
#ifndef MK_MATCH_TORCH_BF16_EPILOGUES
#define MK_MATCH_TORCH_BF16_EPILOGUES 0
#endif
#ifndef MK_FP64_RMS
#define MK_FP64_RMS 0
#endif
#ifndef MK_COMPENSATED_ATTENTION
#define MK_COMPENSATED_ATTENTION 0
#endif
#ifndef MK_FP32_DOWNPROJ
#define MK_FP32_DOWNPROJ 0
#endif
#ifndef MK_ORDERED_DOWNPROJ
#define MK_ORDERED_DOWNPROJ 0
#endif
#ifndef MK_THOR_DYNAMIC_REGISTERS
#define MK_THOR_DYNAMIC_REGISTERS 0
#endif
struct thor_config : default_config {
    static constexpr bool STATIC_REGISTER_ALLOCATION = MK_THOR_DYNAMIC_REGISTERS == 0;
    static constexpr bool ORDERED_DOWNPROJ = MK_ORDERED_DOWNPROJ != 0;
    static constexpr bool COOPERATIVE_LAUNCH = true;
    static constexpr bool FP32_DOWNPROJ = MK_FP32_DOWNPROJ != 0;
    static constexpr bool MATCH_TORCH_BF16_EPILOGUES = MK_MATCH_TORCH_BF16_EPILOGUES != 0;
    static constexpr bool COMPENSATED_ATTENTION = MK_COMPENSATED_ATTENTION != 0;
    static constexpr bool FP64_RMS = MK_FP64_RMS != 0;
    static_assert(!(FP32_DOWNPROJ && ORDERED_DOWNPROJ), "FP32 finalization replaces column ordering");
    static constexpr bool MATCH_TORCH_BF16_RMS = MK_MATCH_TORCH_BF16_RMS != 0;
    static constexpr int INSTRUCTION_BARRIER_PARTIES_PER_WARP = MK_INSTRUCTION_BARRIER_PARTIES;
    static_assert(INSTRUCTION_BARRIER_PARTIES_PER_WARP == 1 || INSTRUCTION_BARRIER_PARTIES_PER_WARP == 32);
    static constexpr int NUM_CONSUMER_WARPS = MK_CONSUMER_WARPS;
    static_assert(NUM_CONSUMER_WARPS == 4 || NUM_CONSUMER_WARPS == 8 || NUM_CONSUMER_WARPS == 16);
    static constexpr int NUM_WARPS = NUM_CONSUMER_WARPS + 4;
    static constexpr int NUM_THREADS = NUM_WARPS * ::kittens::WARP_THREADS;
    static constexpr int CONSUMER_REGISTERS = MK_CONSUMER_REGISTERS;
    static constexpr int NON_CONSUMER_REGISTERS = MK_SERVICE_REGISTERS;
    static_assert(CONSUMER_REGISTERS % 8 == 0 && NON_CONSUMER_REGISTERS % 8 == 0);
    static_assert(CONSUMER_REGISTERS >= 24 && CONSUMER_REGISTERS <= 256);
    static_assert(NON_CONSUMER_REGISTERS >= 24 && NON_CONSUMER_REGISTERS <= 256);
    static constexpr int GMEM_SPIN_LOOP_SLEEP_NANOS = MK_BACKOFF_NS;
    static_assert(GMEM_SPIN_LOOP_SLEEP_NANOS >= 0 && GMEM_SPIN_LOOP_SLEEP_NANOS <= 1000000);
    static constexpr bool TIMING_RECORD_ENABLED = MK_RECORD_TIMING != 0;
    static constexpr int ACTIVE_WORKERS = MK_ACTIVE_WORKERS;
    static_assert(ACTIVE_WORKERS > 0, "active workers must be positive");
    static constexpr int WEIGHT_PIPELINE_STAGES = MK_WEIGHT_PIPELINE_STAGES;
    static constexpr int NUM_PAGES = MK_PAGE_COUNT;
    static_assert(WEIGHT_PIPELINE_STAGES >= 1 && WEIGHT_PIPELINE_STAGES <= 3);
    static_assert(NUM_PAGES >= 1 + 4 * WEIGHT_PIPELINE_STAGES && NUM_PAGES <= 13);
    // Selected page payload plus worst-case 1024-byte alignment loss.
    static constexpr int DYNAMIC_SHARED_MEMORY = NUM_PAGES * PAGE_SIZE + 1023;
};
#endif

template <typename config>
using instruction_layout = kittens::gl<int, 1, -1, -1, config::INSTRUCTION_WIDTH>;
template <typename config>
using timing_layout = kittens::gl<int, 1, -1, -1, config::TIMING_WIDTH>;

template <typename config> void print_config() {
    std::cout << "---------------- CONFIG INFO ----------------" << std::endl;
    std::cout << "INSTRUCTION_PIPELINE_STAGES: "
              << config::INSTRUCTION_PIPELINE_STAGES << std::endl;
    std::cout << "INSTRUCTION_WIDTH: " << config::INSTRUCTION_WIDTH
              << std::endl;
    std::cout << "TIMING_WIDTH: " << config::TIMING_WIDTH << std::endl;
    std::cout << "NUM_CONSUMER_WARPS: " << config::NUM_CONSUMER_WARPS
              << std::endl;
    std::cout << "NUM_WARPS: " << config::NUM_WARPS << std::endl;
    std::cout << "NUM_THREADS: " << config::NUM_THREADS << std::endl;
    std::cout << "NUM_BLOCKS: " << config::NUM_BLOCKS << std::endl;
    std::cout << "CLUSTER_BLOCKS: " << config::CLUSTER_BLOCKS << std::endl;
    std::cout << "MAX_SHARED_MEMORY: " << config::MAX_SHARED_MEMORY
              << std::endl;
    std::cout << "STATIC_SHARED_MEMORY: " << config::STATIC_SHARED_MEMORY
              << std::endl;
    std::cout << "PAGE_SIZE: " << config::PAGE_SIZE << std::endl;
    std::cout << "NUM_PAGES: " << config::NUM_PAGES << std::endl;
    std::cout << "SCRATCH_BYTES: " << config::SCRATCH_BYTES << std::endl;
    std::cout << "DYNAMIC_SEMAPHORES: " << config::DYNAMIC_SEMAPHORES
              << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
}

} // namespace megakernel