#pragma once
#include <cuda/atomic>
#include <stdint.h>
#ifndef MK_PUBLICATION_RELEASE_ONLY
#define MK_PUBLICATION_RELEASE_ONLY 0
#endif

#ifndef MK_PORTABLE_GMEM_BARRIERS
#ifdef MK_THOR
#define MK_PORTABLE_GMEM_BARRIERS 1
#else
#define MK_PORTABLE_GMEM_BARRIERS 0
#endif
#endif

namespace megakernel {
// Every producer completes its global/TMA stores before this RMW. Acq_rel
// RMWs chain producers' publication; an acquire observing the target count
// sees the writes of all preceding producers. No concurrent plain stores/reset
// are permitted: host resets must precede the launch on the same CUDA stream.
template <typename T> __device__ inline T gmem_load_acquire(T *address) {
#if MK_PORTABLE_GMEM_BARRIERS
    return cuda::atomic_ref<T, cuda::thread_scope_device>(*address).load(cuda::memory_order_acquire);
#else
    return *reinterpret_cast<volatile T *>(address);
#endif
}
template <typename T, typename V> __device__ inline T gmem_add_release(T *address, V value) {
#if MK_PORTABLE_GMEM_BARRIERS
    return cuda::atomic_ref<T, cuda::thread_scope_device>(*address).fetch_add(
        static_cast<T>(value),
#if MK_PUBLICATION_RELEASE_ONLY
        // Every producer is a release RMW on the same counter. Later RMWs
        // remain in earlier producers' release sequences; a final acquire
        // observing the target therefore sees every producer's stores.
        cuda::memory_order_release);
#else
        cuda::memory_order_acq_rel);
#endif
#else
    return atomicAdd(address, static_cast<T>(value));
#endif
}

// Debug watchdog is measured in device clock cycles, not cross-SM timestamps.
// The external process watchdog remains required, including for shared waits.
template <typename Config, typename Predicate>
__device__ inline void gmem_wait(Predicate pending) {
#ifdef MK_GMEM_TIMEOUT_CYCLES
    const uint64_t started = clock64();
#endif
    while (pending()) {
#ifdef MK_GMEM_TIMEOUT_CYCLES
        if (clock64() - started > static_cast<uint64_t>(MK_GMEM_TIMEOUT_CYCLES)) {
            printf("Global dependency timeout: worker=%u thread=%u\n", blockIdx.x, threadIdx.x);
            asm volatile("trap;" ::: "memory");
        }
#endif
        __nanosleep(Config::GMEM_SPIN_LOOP_SLEEP_NANOS);
    }
}
} // namespace megakernel
