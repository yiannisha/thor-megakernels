#pragma once
#include <cstddef>

// Format-aware byte accounting for weight stages. Activation, output and
// instruction storage are deliberately separate. NVFP4 scales are canonical
// row-major here; tcgen05's interleaved scale layout needs an explicit physical
// scale allocation, not an assumption that every tile has dense scales.
namespace megakernel::quantized {
constexpr std::size_t align_up(std::size_t n, std::size_t a) {
    return (n + a - 1) / a * a;
}
struct bf16_format {
    static constexpr std::size_t data_bytes(std::size_t n) { return n * 2; }
    static constexpr std::size_t scale_bytes(std::size_t) { return 0; }
};
struct nvfp4_format {
    static constexpr std::size_t data_bytes(std::size_t n) { return (n + 1) / 2; }
    static constexpr std::size_t scale_bytes(std::size_t n) { return (n + 15) / 16; }
};
template<class Format, int Rows, int Columns, int Alignment = 128>
struct weight_stage {
    static_assert(Rows > 0 && Columns > 0 && Columns % 16 == 0);
    static_assert(Alignment > 0 && (Alignment & (Alignment - 1)) == 0);
    static constexpr std::size_t data_bytes = Format::data_bytes(Rows * Columns);
    static constexpr std::size_t scale_offset = align_up(data_bytes, Alignment);
    static constexpr std::size_t scale_bytes = Format::scale_bytes(Rows * Columns);
    static constexpr std::size_t bytes = align_up(scale_offset + scale_bytes, Alignment);
    template<int PageBytes> static constexpr int pages = (bytes + PageBytes - 1) / PageBytes;
};
// Reserve pages independently for the activation buffer, then weight stages.
// This matches the VM's page lifetime ownership without equating a weight page
// with a BF16 tile. The existing allocator supports at most one warp of pages.
template<class Stage, int ActivationBytes, int Stages, int PageBytes>
struct page_plan {
    static_assert(PageBytes >= 128 && PageBytes % 128 == 0);
    static_assert(ActivationBytes > 0 && Stages > 0);
    static constexpr int activation_pages = (ActivationBytes + PageBytes - 1) / PageBytes;
    static constexpr int stage_pages = Stage::template pages<PageBytes>;
    static constexpr int pages = activation_pages + Stages * stage_pages;
    static_assert(pages <= 32, "VM page map exceeds one warp");
    static constexpr int allocated_bytes = pages * PageBytes;
    static constexpr int stage_offset(int i) {
        return (activation_pages + i * stage_pages) * PageBytes;
    }
};
}
