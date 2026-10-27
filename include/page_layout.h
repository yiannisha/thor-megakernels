#pragma once

// Pure arithmetic shared by CUDA and host tests; deliberately has no TK dependency.
#ifdef __CUDACC__
#define MK_LAYOUT_FN static __host__ __device__ constexpr inline
#elif defined(__cplusplus)
#define MK_LAYOUT_FN static constexpr inline
#else
#define MK_LAYOUT_FN static inline
#endif

// Release unused pages first, activation second, then weight stages ordered by
// final consumer use. Each weight stage occupies consecutive physical-page slots.
// Returns -1 for invalid input instead of producing an out-of-bounds page index.
MK_LAYOUT_FN int mk_matvec_release_lid(int pages, int stages, int stage_pages,
                                       int iterations, int query) {
    if (stages < 1 || stage_pages < 1 || pages < 1 + stages * stage_pages ||
        iterations < 0 || query < 0 || query >= pages) return -1;
    const int used_stages = iterations < stages ? iterations : stages;
    const int needed_pages = 1 + used_stages * stage_pages;
    const int unused_pages = pages - needed_pages;
    if (query < unused_pages) return needed_pages + query;
    const int offset = query - unused_pages;
    if (offset == 0) return 0;
    const int first_stage = iterations < stages ? 0 : iterations % stages;
    const int stage = (first_stage + (offset - 1) / stage_pages) % used_stages;
    return 1 + stage * stage_pages + (offset - 1) % stage_pages;
}

// Ops using a contiguous prefix release unused pages then the used prefix.
MK_LAYOUT_FN int mk_prefix_release_lid(int pages, int used_pages, int query) {
    if (pages < 1 || used_pages < 0 || used_pages > pages || query < 0 || query >= pages)
        return -1;
    return (query + used_pages) % pages;
}
#undef MK_LAYOUT_FN
