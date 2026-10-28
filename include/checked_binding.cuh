#pragma once
#include "pyutils/pyutils.cuh"
#include "gmem_barrier.cuh"
#include <stdexcept>
#include <string>

namespace megakernel::py {
inline void check_cuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

// Uses TK conversion, but checks actual compiled resource usage before launch.
// All worker CTAs must be concurrently resident: no oversubscribed spin barriers.
template <auto kernel, typename Config, typename Globals>
void bind_checked_kernel(auto module, auto name, auto Globals::*... members) {
    module.def(name, [](kittens::py::object<decltype(members)>... args,
                        pybind11::kwargs kwargs) -> pybind11::dict {
        Globals g{kittens::py::from_object<
            typename kittens::py::trait<decltype(members)>::member_type>::make(args)...};
        int device;
        check_cuda(cudaGetDevice(&device), "cudaGetDevice");
        cudaDeviceProp prop;
        check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
#ifdef MK_THOR
        if (prop.major != 11 || prop.minor != 0)
            throw std::runtime_error("Thor binary requires compute capability 11.0");
#endif
        const auto grid = g.grid();
        if (grid.x == 0 || grid.y != 1 || grid.z != 1 ||
            grid.x > static_cast<unsigned>(prop.multiProcessorCount))
            throw std::runtime_error("worker grid exceeds the explicitly enabled physical-SM residency budget");
        if (g.instructions.depth() != grid.x || g.timings.depth() != grid.x ||
            g.instructions.rows() != g.timings.rows())
            throw std::runtime_error("instruction/timing queue dimensions do not match worker grid");
#ifdef MK_PHYSICAL_WORKER_IDS
        if (grid.x != prop.multiProcessorCount)
            throw std::runtime_error("physical worker indexing requires full physical-SM queue coverage");
#endif
        int optin;
        check_cuda(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device),
                   "opt-in shared memory attribute");
        cudaFuncAttributes attrs;
        check_cuda(cudaFuncGetAttributes(&attrs, kernel), "cudaFuncGetAttributes");
#ifdef MK_THOR
        if (attrs.numRegs > 96)
            throw std::runtime_error("compiled Thor register usage exceeds the enabling cap");
#endif
        const int dynamic = g.dynamic_shared_memory();
        if (dynamic < Config::NUM_PAGES * Config::PAGE_SIZE + 1023 ||
            dynamic + attrs.sharedSizeBytes > static_cast<size_t>(optin))
            throw std::runtime_error("shared memory payload/alignment/static allocation exceeds budget");
        check_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic),
                   "cudaFuncSetAttribute");
        int resident;
        check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &resident, kernel, Config::NUM_THREADS, dynamic), "occupancy query");
        if (resident < 1)
            throw std::runtime_error("compiled kernel cannot keep one CTA resident per SM");
        if constexpr (requires { Config::NATIVE_TMEM_COLUMNS; }) {
            if (resident != 1)
                throw std::runtime_error("native TMEM ownership requires exactly one resident CTA per SM");
        }
        if constexpr (Config::COOPERATIVE_LAUNCH) {
            if (!prop.cooperativeLaunch || grid.x > static_cast<unsigned>(resident * prop.multiProcessorCount))
                throw std::runtime_error("cooperative grid cannot be admitted with compiled resources");
        }
        pybind11::dict result;
        result["cooperative_launch"] = Config::COOPERATIVE_LAUNCH;
        result["cooperative_launch_supported"] = prop.cooperativeLaunch != 0;
#ifdef MK_BUILD_SOURCE_HASH
        result["build_source_hash"] = MK_BUILD_SOURCE_HASH;
#else
        result["build_source_hash"] = "UNRECORDED";
#endif
        result["portable_gmem_barriers"] = MK_PORTABLE_GMEM_BARRIERS != 0;
        result["physical_sms"] = prop.multiProcessorCount;
        result["resident_worker_capacity"] = resident * prop.multiProcessorCount;
        result["register_mode"] = Config::STATIC_REGISTER_ALLOCATION ? "static" : "dynamic";
        result["consumer_register_target"] = Config::CONSUMER_REGISTERS;
        result["service_register_target"] = Config::NON_CONSUMER_REGISTERS;
        result["consumer_warps"] = Config::NUM_CONSUMER_WARPS;
        result["threads_per_cta"] = Config::NUM_THREADS;
        result["instruction_barrier_parties_per_warp"] = Config::INSTRUCTION_BARRIER_PARTIES_PER_WARP;
        result["backoff_ns"] = Config::GMEM_SPIN_LOOP_SLEEP_NANOS;
        result["timing_enabled"] = Config::TIMING_RECORD_ENABLED;
        result["active_workers"] = grid.x;
        result["page_count"] = Config::NUM_PAGES;
        result["page_size"] = Config::PAGE_SIZE;
        result["resident_ctas_per_sm"] = resident;
        result["static_smem_bytes"] = attrs.sharedSizeBytes;
        result["dynamic_smem_bytes"] = dynamic;
        result["registers_per_thread"] = attrs.numRegs;
        result["local_bytes_per_thread"] = attrs.localSizeBytes;
        if (kwargs.contains("dry_run") && kwargs["dry_run"].cast<bool>()) return result;
        cudaStream_t stream = nullptr;
        if (kwargs.contains("stream"))
            stream = reinterpret_cast<cudaStream_t>(kwargs["stream"].attr("cuda_stream").cast<uintptr_t>());
        if constexpr (Config::COOPERATIVE_LAUNCH) {
            void *parameters[] = {&g};
            check_cuda(cudaLaunchCooperativeKernel(reinterpret_cast<const void *>(kernel), grid, g.block(),
                       parameters, dynamic, stream), "cooperative megakernel launch");
        } else {
            kernel<<<grid, g.block(), dynamic, stream>>>(g);
        }
        check_cuda(cudaGetLastError(), "megakernel launch");
        return result;
    });
}
} // namespace megakernel::py
