#pragma once

#include "llama.cuh"

template <typename Config, kittens::ducks::sv::all sv_t>
__device__ static inline auto
rms_norm(const sv_t &rms_scale_smem, const sv_t &activations_smem,
         float rms_norm_eps, void *scratch_memory) {
    using rv_t = kittens::rv_fl<sv_t::length>;
    rv_t activations_vec, rms_scale_vec;
    if constexpr (Config::FP64_RMS) kittens::warp::sync();
    kittens::warp::load(activations_vec, activations_smem);
    float variance;
    if constexpr (Config::FP64_RMS) {
        static_assert(offsetof(megakernel::instruction_state_t<Config>, scratch) % alignof(double) == 0);
        double partial = 0.;
        for (int i = kittens::laneid(); i < sv_t::length; i += 32) {
            const double value = static_cast<double>(static_cast<float>(activations_smem[i]));
            partial += value * value;
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            partial += __shfl_down_sync(0xffffffff, partial, offset);
        auto *partials = static_cast<double *>(scratch_memory);
        if (kittens::laneid() == 0) partials[kittens::warpid()] = partial;
        kittens::group<Config::NUM_CONSUMER_WARPS>::sync(0);
        variance = 0.f;
        if (kittens::laneid() == 0) {
            double sum = 0.;
            #pragma unroll
            for (int i = 0; i < Config::NUM_CONSUMER_WARPS; ++i) sum += partials[i];
            variance = static_cast<float>(sum / 2048.0);
        }
        variance = __shfl_sync(0xffffffff, variance, 0);
    } else {
        rv_t sq_activations_vec;
        kittens::warp::copy(sq_activations_vec, activations_vec);
        kittens::warp::mul(sq_activations_vec, sq_activations_vec, sq_activations_vec);
        float partial_sum = kittens::warp::sum(sq_activations_vec);
        float *partials = static_cast<float *>(scratch_memory);
        if (kittens::laneid() == 0) partials[kittens::warpid()] = partial_sum;
        kittens::group<Config::NUM_CONSUMER_WARPS>::sync(0);
        float sum = 0.f;
        #pragma unroll
        for (int i = 0; i < Config::NUM_CONSUMER_WARPS; ++i) sum += partials[i];
        variance = sum / 2048.f;
    }
    float rms_scale = rsqrtf(variance + rms_norm_eps);

    kittens::warp::mul(activations_vec, activations_vec, rms_scale);
    // Match the BF16 reference's two rounding points: normalized input,
    // then the learned RMS scale product. Keep the historical float path selectable.
    if constexpr (Config::MATCH_TORCH_BF16_RMS) {
        kittens::rv_bf<sv_t::length> rounded;
        kittens::warp::copy(rounded, activations_vec);
        kittens::warp::copy(activations_vec, rounded);
    }
    kittens::warp::load(rms_scale_vec, rms_scale_smem);
    kittens::warp::mul(activations_vec, activations_vec, rms_scale_vec);
    if constexpr (Config::MATCH_TORCH_BF16_RMS) {
        kittens::rv_bf<sv_t::length> rounded;
        kittens::warp::copy(rounded, activations_vec);
        kittens::warp::copy(activations_vec, rounded);
    }

    return activations_vec;
}

#ifdef KITTENS_BLACKWELL
template <kittens::ducks::st::all st_t>
__device__ static inline void matvec(kittens::sv_fl<st_t::rows> &out_smem,
                                     st_t &weights_smem,
                                     kittens::rv_fl<st_t::cols> &activations) {
    using rt_t = kittens::rt_bf<st_t::rows, st_t::cols>;
    using rrv_t = typename rt_t::row_vec;
    using rcv_t = typename kittens::rt_fl<16, 16>::col_vec;
    // using rcv_t = typename rt_t::col_vec;
    using rv_t = kittens::rv_fl<st_t::rows>;
    using sv_t = kittens::sv_bf<st_t::rows>;

    rrv_t row_activations;
    kittens::warp::copy(row_activations, activations);

    rt_t broadcast_activations, weights;
    kittens::warp::broadcast_col(broadcast_activations, row_activations);
    kittens::warp::load(weights, weights_smem);
    kittens::rt_fl<16, 16> out_activations;
    kittens::warp::zero(out_activations);
    kittens::warp::mma_ABt(out_activations, weights, broadcast_activations,
                  out_activations);
    rcv_t sum_col_vec;
    kittens::warp::row_max(sum_col_vec, out_activations);

    rv_t sum_vec;
    kittens::warp::copy(sum_vec, sum_col_vec);

    if (kittens::laneid() < 16) {
        out_smem[kittens::laneid()] = sum_vec[0][0];
    }
    kittens::warp::sync();
}
#else
template <kittens::ducks::st::all st_t>
__device__ static inline void matvec(kittens::sv_fl<st_t::rows> &out_smem,
                                     st_t &weights_smem,
                                     kittens::rv_fl<st_t::cols> &activations) {
    using rt_t = kittens::rt_fl<st_t::rows, st_t::cols>;
    using rrv_t = typename rt_t::row_vec;
    using rcv_t = typename rt_t::col_vec;
    using rv_t = kittens::rv_fl<st_t::rows>;
    using sv_t = kittens::sv_bf<st_t::rows>;

    rrv_t row_activations;
    kittens::warp::copy(row_activations, activations);

    rt_t broadcast_activations, weights;
    kittens::warp::broadcast_col(broadcast_activations, row_activations);
    kittens::warp::load(weights, weights_smem);
    kittens::warp::mul(broadcast_activations, broadcast_activations, weights);
    rcv_t sum_col_vec;
    kittens::warp::row_sum(sum_col_vec, broadcast_activations);

    rv_t sum_vec;
    kittens::warp::copy(sum_vec, sum_col_vec);

    if (kittens::laneid() < 16) {
        out_smem[kittens::laneid()] = sum_vec[0][0];
    }
    kittens::warp::sync();
}
#endif

template <typename Config, kittens::ducks::sv::all sv_t, typename rv_t,
          int SCRATCH_BYTES_PER_WARP>
__device__ static inline void matvec_reduce(uint8_t *scratch, rv_t &sum_vec) {
    rv_t part_vec;
    kittens::warp::zero(sum_vec);

#pragma unroll
    for (int i = 0; i < Config::NUM_CONSUMER_WARPS; i++) {

        // TODO: for now, deliberately not using sizeof(sv_t) here because we've
        // had alignment issues before.
        sv_t &part =
            *reinterpret_cast<sv_t *>(scratch + (i * SCRATCH_BYTES_PER_WARP));

        kittens::warp::load(part_vec, part);
        kittens::warp::add(sum_vec, sum_vec, part_vec);
    }
}

