"""Prepare isolated BF16 source adaptations; never modify either upstream."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
FLASH = ROOT.parent / "FlashRT"
OUT = ROOT / "local/pi0/faithful/sources"
FLASH_REV = "86ef457d16ce5ea490d94493991e76078fcaa97d"
CUTLASS_REV = "da5e086dab31d63815acafdac9a9c5893b1c69e2"


def prepare():
    revision = subprocess.check_output(["git", "-C", str(FLASH), "rev-parse", "HEAD"], text=True).strip()
    if revision != FLASH_REV:
        raise RuntimeError(f"Expected FlashRT {FLASH_REV}, found {revision}")
    cutlass_revision = subprocess.check_output(["git", "-C", str(ROOT / "local/pi0/faithful/cutlass"), "rev-parse", "HEAD"], text=True).strip()
    if cutlass_revision != CUTLASS_REV:
        raise RuntimeError(f"Expected CUTLASS {CUTLASS_REV}, found {cutlass_revision}")
    OUT.mkdir(parents=True, exist_ok=True)
    source = FLASH / "csrc/gemm/gemm_types_sm100_fp16.h"
    original = source.read_text()
    needle = "using cutlass_fp16_t = cutlass::half_t;"
    if original.count(needle) != 1:
        raise RuntimeError("FlashRT GEMM element alias changed")
    adapted = original.replace('#include "cutlass/cutlass.h"', '#include "cutlass/cutlass.h"\n#include "cutlass/bfloat16.h"')
    adapted = adapted.replace(needle, "using cutlass_fp16_t = cutlass::bfloat16_t;")
    (OUT / "gemm_types_bf16.h").write_text(adapted)
    # The original two-phase kernel and visitor include overlay are unchanged.
    source2 = FLASH / "csrc/gemm/mega/flashrt_megakernel_geglu_g8.cu"
    original2 = source2.read_text()
    adapted2 = original2.replace('#include "cutlass/half.h"', '#include "cutlass/bfloat16.h"')
    adapted2 = adapted2.replace("using fp16_t = cutlass::half_t;", "using fp16_t = cutlass::bfloat16_t;")
    adapted2 = adapted2.replace("__half", "__nv_bfloat16").replace("rms_norm_fp16", "rms_norm")
    adapted2 = adapted2.replace("cutlass_fp16_2sm21", "faithful_bf16_2sm21").replace("cutlass_fp16_k64", "faithful_bf16_k64")
    adapted2 = adapted2.replace("flashrt_megakernel_geglu_g8_fp16", "faithful_bf16_geglu_down")
    adapted2 = adapted2.replace("flashrt_rms_qkv_fp16", "faithful_bf16_rms_qkv")
    adapted2 = adapted2.replace("flashrt_encoder_ffn_block_fp16", "faithful_bf16_ffn_block")
    (OUT / "geglu_bf16.cu").write_text(adapted2)
    # Preserve source reduction trees, rotary pair arithmetic and patch order.
    def bf16(text):
        return (text.replace("__half2float", "__bfloat162float")
                .replace("__float2half", "__float2bfloat16")
                .replace("__half2", "__nv_bfloat162")
                .replace("__half", "__nv_bfloat16")
                .replace("_fp16", "_bf16"))
    kernels = FLASH / "csrc/kernels"
    softmax = (kernels / "softmax.cu").read_text().replace('#include "softmax.cuh"', '#include <cuda_runtime.h>\n#include <cuda_bf16.h>')
    (OUT / "softmax_bf16.cu").write_text(bf16(softmax))
    rope = (kernels / "rope.cu").read_text()
    rope = rope[rope.index("__global__ void qkv_split_rope_kvcache_fp16_kernel"):rope.index("__global__ void qkv_split_rope_kvcache_fp16_devpos_kernel")]
    (OUT / "rope_bf16.cu").write_text('#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n' + bf16(rope))
    patch = (kernels / "patch_embed.cu").read_text()
    patch = patch[patch.index("__global__ void patch_im2col_kernel"):patch.index("__global__ void patch_im2col_uint8_kernel")]
    import re
    patch = re.sub(r"\bhalf\b", "__nv_bfloat16", patch).replace("patch_im2col", "faithful_patch_im2col")
    (OUT / "patch_bf16.cu").write_text('#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n' + patch)
    patch_source = (kernels / "patch_embed.cu").read_text()
    pointwise = patch_source[patch_source.index("__global__ void patch_embed_bias_pos_kernel"):patch_source.index("// ── bf16 NCHW")]
    pointwise = re.sub(r"\bhalf\b", "__nv_bfloat16", bf16(pointwise)).replace("patch_embed_bias_pos", "faithful_patch_bias_pos_impl")
    (OUT / "patch_bias_bf16.cu").write_text('#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n'+pointwise)
    fmha = (FLASH / "csrc/attention/fmha_fp16_strided.cu").read_text()
    fmha = fmha.replace("cutlass::half_t", "cutlass::bfloat16_t").replace("fmha_fp16", "faithful_fmha_bf16")
    (OUT / "fmha_bf16.cu").write_text(fmha)
    # VM-only overlay: source collectives retain all pipeline and epilogue
    # logic, while the enclosing VM owns the one lifetime TMEM allocation.
    kernel_path = "cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp"
    kernel = (ROOT / "local/pi0/faithful/cutlass/include" / kernel_path).read_text()
    changes = {
        "operator() (Params const& params, char* smem_buf)": "operator() (Params const& params, char* smem_buf, uint32_t vm_tmem_base = 0)",
        "tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);": "if (cute::elect_one_sync()) shared_storage.tmem_base_ptr = vm_tmem_base;",
        "tmem_allocator.release_allocation_lock();": "/* allocation permit belongs to the VM */",
        "tmem_allocator.free(tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);": "/* allocation is released when the VM exits */",
        "uint32_t lane_predicate = cute::elect_one_sync();": "if (warp_idx >= 8) warp_category = static_cast<WarpCategory>(-1);\n    uint32_t lane_predicate = cute::elect_one_sync();",
        "pipeline_init_wait(cluster_size);": "pipeline_init_wait(cluster_size);\n    if (!work_tile_info.is_valid()) return;",
    }
    for before, after in changes.items():
        if kernel.count(before) != 1:
            raise RuntimeError(f"VM CUTLASS adapter anchor changed: {before}")
        kernel = kernel.replace(before, after)
    overlay = OUT / "vm" / kernel_path
    overlay.parent.mkdir(parents=True, exist_ok=True)
    overlay.write_text(kernel)
    geglu_kernel_name = "flashrt_megakernel_geglu_g8_kernel.hpp"
    geglu_kernel = (FLASH / "csrc/gemm/mega" / geglu_kernel_name).read_text()
    for before, after in changes.items():
        if geglu_kernel.count(before) != 1:
            raise RuntimeError(f"VM GeGLU adapter anchor changed: {before}")
        geglu_kernel = geglu_kernel.replace(before, after)
    (OUT / "vm" / geglu_kernel_name).write_text(geglu_kernel)
    native_geglu_types = adapted2[:adapted2.index("// Forward decls for the production kernels")]
    native_geglu_types = native_geglu_types.replace("    void>;", "    faithful::StaticSchedulerTag>;")
    native_geglu_types = native_geglu_types.replace("namespace {", "namespace faithful_geglu {")
    native_geglu_types += "\nusing GemmKernel=faithful_geglu::GemmKernel;\nusing GemmOp=faithful_geglu::GemmOp;\n"
    (OUT / "geglu_native_types.h").write_text(native_geglu_types)
    fmha_kernel_name = "kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp"
    fmha_kernel = (FLASH / "csrc/attention" / fmha_kernel_name).read_text()
    fmha_changes = {
        "operator()(const Params &params, char* smem)": "operator()(const Params &params, char* smem, uint32_t vm_tmem_base, int vm_tile)",
        "TileScheduler tile_scheduler{params.tile_scheduler};": "TileScheduler tile_scheduler{params.tile_scheduler, vm_tile};",
        "tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);": "if (cute::elect_one_sync()) shared_storage.tmem_base_ptr = vm_tmem_base;",
    }
    for before, after in fmha_changes.items():
        if fmha_kernel.count(before) != 1:raise RuntimeError(("FMHA adapter anchor",before))
        fmha_kernel = fmha_kernel.replace(before,after)
    fmha_kernel = fmha_kernel.replace("tmem_allocator.free(free_stage_ptr, TmemAllocator::Sm100TmemCapacityColumns);", "/* VM owns TMEM lifetime */")
    # Bring up one safe uniform 128-register policy first. Source per-role
    # redistribution needs a separate VM-state handoff before it can be enabled.
    fmha_kernel = re.sub(r"(?:cutlass::arch::)?warpgroup_reg_(?:set|dealloc)<[^>]+>\(\);", "/* static VM register policy */", fmha_kernel)
    fmha_overlay = OUT / "vm" / fmha_kernel_name
    fmha_overlay.parent.mkdir(parents=True,exist_ok=True)
    fmha_overlay.write_text(fmha_kernel)
    fmha_types = fmha[:fmha.index("static void* g_ws")]
    fmha_types = '#include "fmha_scheduler.cuh"\n'+fmha_types
    fmha_types = fmha_types.replace("cutlass::fmha::kernel::IndividualTileScheduler", "faithful::FmhaScheduler")
    (OUT / "fmha_native_types.h").write_text(fmha_types)
    from cluster_overlay import prepare_cluster_overlay
    prepare_cluster_overlay(ROOT / "local/pi0/faithful/cutlass/include", OUT / "vm")
    from pointwise_prepare import prepare_pointwise
    prepare_pointwise(FLASH, OUT)
    down_start = adapted.index("namespace sm100_fp16_2sm21 {")
    down_end = adapted.index("}  // namespace sm100_fp16_2sm21",down_start)+len("}  // namespace sm100_fp16_2sm21")
    coherent_down = adapted[down_start:down_end].replace("sm100_fp16_2sm21", "coherent_down")
    coherent_down = coherent_down.replace("Shape<_2, _1, _1>", "Shape<_2, _2, _1>")
    (OUT / "coherent_down_types.h").write_text(coherent_down)
    manifest = {
        "flashrt_revision": revision,
        "cutlass_revision": cutlass_revision,
        "adaptation": "BF16 operands/output/auxiliary storage; FP32 accumulation; source tactics/EVTs preserved",
        "upstream": {str(p.relative_to(FLASH)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, source2)},
        "generated": {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob("*") if p.suffix in (".h", ".hpp", ".cuh", ".cu")},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    prepare()
