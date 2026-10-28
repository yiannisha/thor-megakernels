"""Record isolated probe provenance without touching production build guards."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    binary=Path(sys.argv[1]).resolve()
    files=[ROOT/'tools/analysis/nvfp4/primitive.cu',ROOT/'experimental/quantized_runtime/persistent.cu',
           ROOT/'experimental/quantized_runtime/format_storage.cuh']
    if not binary.name.startswith('persistent'): files=files[:1]
    report=dict(binary_sha256=sha(binary),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
                cutlass_revision=subprocess.check_output(['git','-C',str(ROOT/'local/cutlass-nvfp4'),'rev-parse','HEAD'],text=True).strip(),
                cuda_arch='sm_110a',stage_argument=sys.argv[2] if len(sys.argv)>2 else 'auto')
    overlay=ROOT/'local/thor-nvfp4/overlay/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp'
    if binary.name.startswith('persistent'):
        report['overlay_sha256']=sha(overlay)
        report['overlay_generator_sha256']=sha(ROOT/'tools/analysis/nvfp4/cutlass_overlay.py')
    Path(str(binary)+'.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
