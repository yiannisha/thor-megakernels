#pragma once

#include "llama.cuh"
#include "utils.cuh"
#include "matvec_pipeline.cuh"

using namespace kittens;
using namespace megakernel;

template <int _EXPECTED_ARRIVAL_COUNT, auto WeightsPtr,
          auto InputActivationsPtr, auto OutputActivationsPtr, int _opcode,
          int _prev_opcode = 0,
          typename Config = default_config,
          typename Globals = llama_1b_globals>

struct MatVecAddOp {
    static constexpr int opcode = _opcode;
    static constexpr int prev_opcode = _prev_opcode;
    static constexpr int EXPECTED_ARRIVAL_COUNT = _EXPECTED_ARRIVAL_COUNT;

    struct parsed_instruction {
        int layer, start_block_idx, end_block_idx, reduction_block_idx,
            start_reduction_col, iters;
        __device__ inline parsed_instruction(
            typename Config::instruction_t &instruction) {
            layer = instruction[1]; // in units of 1
            start_block_idx =
                instruction[2]; // in units of 1 (0, 16, 32, ..., 2032)
            end_block_idx =
                instruction[3]; // in units of 1 (0, 16, 32, ..., 2032)
            reduction_block_idx = instruction[4]; // in units of hidden_dim=2048
                                                  // (0, 2048, 4096, 6144)
            start_reduction_col = reduction_block_idx * Globals::hidden_dim;
            iters = end_block_idx - start_block_idx;
        }
        __device__ inline parsed_instruction(megakernel::state<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    struct pipeline_specifics {

        static __device__ inline void
        load_iter(megakernel::state<Config> &s, const globals &g, parsed_instruction &inst,
                  int iter, int col_idx, kittens::st_bf<16, Config::WEIGHT_CHUNK_WIDTH> &weight_chunk,
                  kittens::semaphore &sem) {
            kittens::tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(
                weight_chunk, g.*WeightsPtr,
                coord<>{inst.layer,
                        (inst.start_block_idx + iter) *
                            Globals::matvec_block_size,
                        inst.start_reduction_col + Config::WEIGHT_CHUNK_WIDTH * col_idx},
                sem);
        }

        static __device__ inline void store(megakernel::state<Config> &s, const globals &g,
                                            parsed_instruction &inst,
                                            int output_idx, int output_stage) {

            int block_idx = inst.start_block_idx + output_idx;

            uint8_t *output_scratch_start =
                pipeline::get_output_start(s, output_stage);
            kittens::sv_bf<16> &output_smem_bf =
                *reinterpret_cast<kittens::sv_bf<16> *>(output_scratch_start);

            kittens::rv_fl<16> output_rv;
            matvec_reduce<Config, kittens::sv_fl<16>, kittens::rv_fl<16>,
                          pipeline::SCRATCH_BYTES_PER_WARP>(
                output_scratch_start, output_rv);

            if constexpr (opcode == OPCODE_DownProjResidual && Config::FP32_DOWNPROJ) {
                // Each (column, output element) has exactly one writer. The
                // last of four column arrivals finalizes this output block;
                // no CTA waits for work that has not been scheduled yet.
                static_assert(Globals::intermediate_dim == 4 * Globals::hidden_dim);
                const int lane = kittens::warp::laneid();
                const int index = block_idx * Globals::matvec_block_size + lane;
                if (lane < Globals::matvec_block_size) {
                    g.down_partials.raw_ptr[inst.reduction_block_idx * Globals::hidden_dim + index] = output_rv[0][0];
                    __threadfence();
                }
                kittens::warp::sync();
                unsigned int prior = 0;
                if (lane == 0) {
                    // Slots 7..9 are reserved for per-block finalization;
                    // ordinary opcodes use slots 0..6 only.
                    prior = gmem_add_release(&g.Bar[{inst.layer, 7 + block_idx / 48, block_idx % 48}], 1);
                }
                prior = __shfl_sync(0xffffffff, prior, 0);
                kittens::warp::sync();
                if (prior == 3) {
                    if (lane < Globals::matvec_block_size) {
                        float sum = 0.f;
                        #pragma unroll
                        for (int col = 0; col < 4; ++col)
                            sum += g.down_partials.raw_ptr[col * Globals::hidden_dim + index];
                        // PyTorch rounds the linear output to BF16 before
                        // adding the BF16 residual, then rounds the result.
                        const float projected = __bfloat162float(__float2bfloat16_rn(sum));
                        const float residual = __bfloat162float(g.hidden_states.raw_ptr[index]);
                        g.hidden_states.raw_ptr[index] = __float2bfloat16_rn(projected + residual);
                        __threadfence();
                    }
                    kittens::warp::sync();
                    if (lane == 0)
                        gmem_add_release(&g.Bar[{inst.layer, opcode - 1, 0}], 4);
                }
                return;
            }

            kittens::warp::sync();
            kittens::warp::store(output_smem_bf, output_rv);
            kittens::warp::sync();

            if (kittens::warp::laneid() == 0) {
                auto &OutputActivations =
                    g.*OutputActivationsPtr; // object in global memory
                kittens::tma::store_add_async<cache_policy::EVICT_LAST>(
                    OutputActivations, output_smem_bf, {block_idx});
                kittens::tma::store_async_read_wait();
            }

            kittens::warp::sync();
        }
    };
    using pipeline = matvec_pipeline<Config, Globals, parsed_instruction,
                                     pipeline_specifics>;

    struct controller {
        static __device__ int
        release_lid(const Globals &g,
                    typename Config::instruction_t &instruction, int &query) {
            return pipeline::release_lid(g, instruction, query);
        }

        static __device__ int init_semaphores(const Globals &g,
                                              megakernel::state<Config> &s) {
            return pipeline::init_semaphores(s);
        }
    };
    struct loader {
        static __device__ void run(const Globals &g, megakernel::state<Config> &s) {
            pipeline::loader_loop(s, g);
        }
    };
    struct launcher {
        static __device__ void run(const globals &g, megakernel::state<Config> &s) {
            if (kittens::laneid() == 0) {
#ifdef KITTENS_BLACKWELL
                s.wait_tensor_ready();
                arrive(s.tensor_finished, Config::NUM_CONSUMER_WARPS);
#endif
            }
        }
    };
    struct consumer {
        static __device__ void run(const Globals &g, megakernel::state<Config> &s) {

            using sv_t = kittens::sv_bf<pipeline::REDUCTION_DIM_PER_WARP>;
            using rv_t = kittens::rv_fl<pipeline::REDUCTION_DIM_PER_WARP>;
            parsed_instruction inst{s};

            if (kittens::laneid() == 0 && kittens::warpid() == 0) {

                int activation_page = pipeline::get_activation_page(s);
                s.wait_page_ready(activation_page);

                s.record(megakernel::TEVENT_AT_GMEM_WAIT);
                if constexpr (opcode == OPCODE_DownProjResidual && Config::ORDERED_DOWNPROJ) {
                    // Preserve the PyVM's column order for BF16 residual updates.
                    // Each completed column contributes exactly hidden_dim/16 tasks.
                    megakernel::gmem_wait<Config>([&] {
                        return gmem_load_acquire(&g.Bar[{inst.layer, opcode - 1, 0}]) <
                            inst.reduction_block_idx * (Globals::hidden_dim / Globals::matvec_block_size);
                    });
                }
                megakernel::gmem_wait<Config>([&] { return (gmem_load_acquire(&g.Bar[{inst.layer, prev_opcode - 1,
                                                inst.reduction_block_idx}]) <
                       EXPECTED_ARRIVAL_COUNT); });
                s.record(megakernel::TEVENT_DONE_GMEM_WAIT);

                auto &activations = pipeline::get_activations(s);
                auto &InputActivations =
                    g.*InputActivationsPtr; // object in global memory
            }
            group<Config::NUM_CONSUMER_WARPS>::sync(4);

            sv_t &activations_smem = reinterpret_cast<sv_t *>(
                &pipeline::get_activations(s))[kittens::warpid()];

            kittens::warp::load(activations_smem, g.*InputActivationsPtr,
                       coord<>{inst.start_reduction_col +
                               kittens::warpid() * pipeline::REDUCTION_DIM_PER_WARP});
            kittens::warp::sync();

            rv_t activations_vec;
            kittens::warp::load(activations_vec, activations_smem);
            kittens::warp::sync();

            s.warp_finish_page(pipeline::get_activation_page(s), 1);

            pipeline::consumer_loop(s, g, activations_vec);
        }
    };
    struct storer {
        // Uses 4 full pages for outputs.
        static __device__ void run(const globals &g, megakernel::state<Config> &s) {
            pipeline::storer_loop(s, g);
            kittens::warp::sync();

            if (kittens::laneid() == 0) {
                s.record(megakernel::TEVENT_AT_GMEM_STORE);
                parsed_instruction inst{s};

                kittens::tma::store_async_wait(); // not just read wait! full wait! must
                                         // be visible in global!

                // asm volatile("fence.acq_rel.gpu;\n"); // possible we need sc
                // here but I don't think so.
                if constexpr (!(opcode == OPCODE_DownProjResidual && Config::FP32_DOWNPROJ))
                    gmem_add_release(&g.Bar[{inst.layer, opcode - 1, 0}], inst.iters);
                s.record(megakernel::TEVENT_DONE_GMEM_STORE);
            }
        }
    };
};

template <typename Config, typename Globals>
struct downproj : MatVecAddOp<llama_1b_globals::hidden_dim /
                                  llama_1b_globals::matvec_block_size,
                              &Globals::down_weights, &Globals::silu_out,
                              &Globals::hidden_states, OPCODE_DownProjResidual,
                              OPCODE_DownProjResidual - 1, Config, Globals> {};

template <typename Config, typename Globals>
struct o_proj : MatVecAddOp<llama_1b_globals::num_attention_heads,
                            &Globals::o_weights, &Globals::attn_out,
                            &Globals::hidden_states, OPCODE_O_ProjResidual,
                            OPCODE_O_ProjResidual - 1, Config, Globals> {};

