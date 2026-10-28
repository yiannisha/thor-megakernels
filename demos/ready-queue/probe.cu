// Model-free ready-only scheduler prototype. Does not run Llama or OpenPI.
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cooperative_groups.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_OK(call) do { auto s = (call); if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); } while (0)
using Atomic = cuda::atomic_ref<int, cuda::thread_scope_device>;
struct Device {
    int *parents, *offsets, *children, *pending, *done, *executions;
    int *queue, *published, *head, *tail, *completed;
    float *input, *output;
};

__device__ int take_ready(Device d) {
    int position = Atomic(*d.head).load(cuda::memory_order_relaxed);
    for (;;) {
        if (position >= Atomic(*d.tail).load(cuda::memory_order_acquire)) return -1;
        if (Atomic(*d.head).compare_exchange_weak(position, position + 1, cuda::memory_order_relaxed)) break;
    }
    // The reserving producer is a resident worker. A slot is published once;
    // total enqueues never exceed task count, so there is no ring wrap/ABA.
    while (!Atomic(d.published[position]).load(cuda::memory_order_acquire)) __nanosleep(100);
    return d.queue[position];
}

__device__ void publish_ready(Device d, int task) {
    int position = Atomic(*d.tail).fetch_add(1, cuda::memory_order_relaxed);
    d.queue[position] = task;
    Atomic(d.published[position]).store(1, cuda::memory_order_release);
}

template<bool Dynamic>
__global__ void execute(Device d, int tasks, int width, int delay_ns) {
    cooperative_groups::this_grid().sync();
    __shared__ int task;
    int cursor = blockIdx.x;
    for (;;) {
        if (threadIdx.x == 0) {
            if constexpr (Dynamic) {
                do {
                    task = take_ready(d);
                    if (task < 0 && Atomic(*d.completed).load(cuda::memory_order_acquire) == tasks) break;
                    if (task < 0) __nanosleep(100);
                } while (task < 0);
            } else {
                task = cursor < tasks ? cursor : -1;
                if (task >= 0) {
                    for (int p = 0; p < 2; ++p) {
                        int parent = d.parents[task * 2 + p];
                        if (parent >= 0)
                            while (!Atomic(d.done[parent]).load(cuda::memory_order_acquire)) __nanosleep(100);
                    }
                }
                cursor += gridDim.x;
            }
            if (task >= 0) {
                Atomic(d.executions[task]).fetch_add(1, cuda::memory_order_relaxed);
                if (delay_ns) __nanosleep(((task * 17) % 41) * delay_ns);
            }
        }
        __syncthreads();
        if (task < 0) return;
        for (int i = threadIdx.x; i < width; i += blockDim.x) {
            float value = .5f * d.input[i] + .001f * (task % 97);
            int p0 = d.parents[task * 2], p1 = d.parents[task * 2 + 1];
            if (p0 >= 0) value += .25f * d.output[p0 * width + i];
            if (p1 >= 0) value += .125f * d.output[p1 * width + i];
            d.output[task * width + i] = value;
        }
        // Every writer fences before the leader publishes completion. Each
        // child's acq_rel decrement chains all parents into its queue release.
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
            Atomic(d.done[task]).store(1, cuda::memory_order_release);
            if constexpr (Dynamic) {
                for (int i = d.offsets[task]; i < d.offsets[task + 1]; ++i) {
                    int child = d.children[i];
                    if (Atomic(d.pending[child]).fetch_sub(1, cuda::memory_order_acq_rel) == 1)
                        publish_ready(d, child);
                }
                Atomic(*d.completed).fetch_add(1, cuda::memory_order_release);
            }
        }
        __syncthreads();
    }
}

template<class T> T *allocate(size_t count) {
    T *pointer;
    CUDA_OK(cudaMalloc(&pointer, std::max(size_t(1), count) * sizeof(T)));
    return pointer;
}
template<class T> void copy(T *destination, const std::vector<T> &source) {
    if (!source.empty()) CUDA_OK(cudaMemcpy(destination, source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice));
}

int main(int argc, char **argv) try {
    if (argc != 7 && argc != 8) throw std::runtime_error("usage: probe tasks workers seed random|chain|wide delay_ns repeats [static|ready|both]");
    int tasks = std::stoi(argv[1]), workers = std::stoi(argv[2]), seed = std::stoi(argv[3]);
    std::string shape = argv[4];
    int delay = std::stoi(argv[5]), repeats = std::stoi(argv[6]);
    std::string selected = argc == 8 ? argv[7] : "both";
    constexpr int width = 256, threads = 128;
    cudaDeviceProp prop; CUDA_OK(cudaGetDeviceProperties(&prop, 0));
    if (tasks < 1 || tasks > 65536 || workers < 1 || workers > prop.multiProcessorCount ||
        delay < 0 || delay > 1000 || repeats < 1 || repeats > 1000 ||
        (selected != "static" && selected != "ready" && selected != "both") ||
        (shape != "random" && shape != "chain" && shape != "wide") || !prop.cooperativeLaunch)
        throw std::runtime_error("invalid configuration or cooperative launch unavailable");
    std::mt19937 rng(seed);
    std::vector<int> parents(tasks * 2, -1), pending(tasks, 0), offsets(tasks + 1), children;
    std::vector<std::vector<int>> outgoing(tasks);
    for (int task = 1; task < tasks; ++task) {
        if (shape == "chain") parents[task * 2] = task - 1;
        else if (shape == "random" && task % 5) {
            parents[task * 2] = rng() % task;
            if (task > 2 && task % 3 == 0) {
                int second = rng() % task;
                if (second != parents[task * 2]) parents[task * 2 + 1] = second;
            }
        }
        for (int p = 0; p < 2; ++p) if (parents[task * 2 + p] >= 0) {
            ++pending[task]; outgoing[parents[task * 2 + p]].push_back(task);
        }
    }
    for (int task = 0; task < tasks; ++task) {
        offsets[task] = children.size();
        children.insert(children.end(), outgoing[task].begin(), outgoing[task].end());
    }
    offsets[tasks] = children.size();
    std::vector<int> queue(tasks, -1), published(tasks, 0);
    int roots = 0;
    for (int task = 0; task < tasks; ++task) if (pending[task] == 0) {
        queue[roots] = task; published[roots++] = 1;
    }
    std::vector<float> input(width), reference(size_t(tasks) * width), actual(reference.size());
    for (int i = 0; i < width; ++i) input[i] = (int(rng() % 201) - 100) / 100.f;
    for (int task = 0; task < tasks; ++task) for (int i = 0; i < width; ++i) {
        float value = .5f * input[i] + .001f * (task % 97);
        if (parents[task * 2] >= 0) value += .25f * reference[parents[task * 2] * width + i];
        if (parents[task * 2 + 1] >= 0) value += .125f * reference[parents[task * 2 + 1] * width + i];
        reference[task * width + i] = value;
    }
    Device d{allocate<int>(parents.size()), allocate<int>(offsets.size()), allocate<int>(children.size()),
             allocate<int>(tasks), allocate<int>(tasks), allocate<int>(tasks), allocate<int>(tasks), allocate<int>(tasks),
             allocate<int>(1), allocate<int>(1), allocate<int>(1), allocate<float>(width), allocate<float>(reference.size())};
    copy(d.parents, parents); copy(d.offsets, offsets); copy(d.children, children); copy(d.input, input);
    std::vector<int> executions(tasks);
    cudaEvent_t start, stop; CUDA_OK(cudaEventCreate(&start)); CUDA_OK(cudaEventCreate(&stop));
    std::printf("{\"scope\":\"model-free FP32 vector DAG; not Llama/OpenPI\",\"tasks\":%d,\"workers\":%d,\"seed\":%d,\"shape\":\"%s\",\"delay_ns\":%d,\"width\":%d,\"warmups\":5,\"repeats\":%d,\"batches\":3,\"variants\":[", tasks, workers, seed, shape.c_str(), delay, width, repeats);
    bool first_mode = true;
    for (int mode = 0; mode < 2; ++mode) {
        if ((selected == "static" && mode) || (selected == "ready" && !mode)) continue;
        auto kernel = mode ? execute<true> : execute<false>;
        int resident; CUDA_OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, threads, 0));
        if (resident < 1 || workers > resident * prop.multiProcessorCount) throw std::runtime_error("resident grid admission failed");
        cudaFuncAttributes attrs; CUDA_OK(cudaFuncGetAttributes(&attrs, kernel));
        std::vector<float> timings;
        float maximum = 0;
        for (int repeat = -5; repeat < repeats * 3; ++repeat) {
            copy(d.pending, pending); copy(d.queue, queue); copy(d.published, published);
            CUDA_OK(cudaMemset(d.done, 0, tasks * sizeof(int))); CUDA_OK(cudaMemset(d.executions, 0, tasks * sizeof(int)));
            CUDA_OK(cudaMemset(d.head, 0, sizeof(int))); CUDA_OK(cudaMemset(d.completed, 0, sizeof(int)));
            CUDA_OK(cudaMemcpy(d.tail, &roots, sizeof(int), cudaMemcpyHostToDevice));
            CUDA_OK(cudaMemset(d.output, 0, reference.size() * sizeof(float)));
            void *args[] = {&d, &tasks, const_cast<int *>(&width), &delay};
            CUDA_OK(cudaEventRecord(start));
            CUDA_OK(cudaLaunchCooperativeKernel(reinterpret_cast<void *>(kernel), workers, threads, args, 0));
            CUDA_OK(cudaEventRecord(stop)); CUDA_OK(cudaEventSynchronize(stop));
            float elapsed; CUDA_OK(cudaEventElapsedTime(&elapsed, start, stop));
            CUDA_OK(cudaMemcpy(actual.data(), d.output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_OK(cudaMemcpy(executions.data(), d.executions, tasks * sizeof(int), cudaMemcpyDeviceToHost));
            for (int task = 0; task < tasks; ++task) if (executions[task] != 1) throw std::runtime_error("task coverage failure");
            for (size_t i = 0; i < actual.size(); ++i) {
                if (!std::isfinite(actual[i])) throw std::runtime_error("nonfinite result");
                maximum = std::max(maximum, std::abs(actual[i] - reference[i]));
            }
            if (maximum > 1e-5f) throw std::runtime_error("reference mismatch");
            if (repeat >= 0) timings.push_back(elapsed);
        }
        std::printf("%s{\"mode\":\"%s\",\"status\":\"PASS\",\"max_abs\":%.9g,\"registers_per_thread\":%d,\"resident_ctas_per_sm\":%d,\"samples_ms\":[", first_mode ? "" : ",", mode ? "ready_queue" : "static_rr", maximum, attrs.numRegs, resident);
        for (size_t i = 0; i < timings.size(); ++i) std::printf("%s%.9g", i ? "," : "", timings[i]);
        std::printf("]}");
        first_mode = false;
    }
    std::printf("]}\n");
    for (void *pointer : std::vector<void *>{d.parents, d.offsets, d.children, d.pending, d.done,
                                            d.executions, d.queue, d.published, d.head, d.tail,
                                            d.completed, d.input, d.output}) CUDA_OK(cudaFree(pointer));
    CUDA_OK(cudaEventDestroy(start)); CUDA_OK(cudaEventDestroy(stop));
    return 0;
} catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
}
