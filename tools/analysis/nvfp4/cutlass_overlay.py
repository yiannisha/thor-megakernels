"""Generate the narrow CUTLASS reentry patch; upstream checkout stays clean.

Standalone launches keep their original permit lifetime. Only cooperative
runtime descriptors retain the allocation permit until the instruction loop ends.
"""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def main():
    relative=Path('cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp')
    source=(ROOT/'local/cutlass-nvfp4/include'/relative).read_text()
    changes={
        '    KernelHardwareInfo hw_info{}; \n  };':
        '    KernelHardwareInfo hw_info{}; \n    bool retain_tmem_permit = false;\n  };',
        '      tmem_allocator.release_allocation_lock();':
        '      if (!params.retain_tmem_permit) tmem_allocator.release_allocation_lock();',
    }
    for old,new in changes.items():
        if source.count(old)!=1: raise RuntimeError(f'CUTLASS overlay anchor changed: {old!r}')
        source=source.replace(old,new)
    target=ROOT/'local/thor-nvfp4/overlay'/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists() or target.read_text()!=source: target.write_text(source)
if __name__=='__main__':main()
