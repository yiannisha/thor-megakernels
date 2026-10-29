"""Prepare the pinned vision arithmetic for a 128-thread VM consumer group."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
REV='6c4f74fb338e0c3cdb07ac6f5eab5f54fc367c15'
CUTLASS_REV='91843b71d90737e0a9b86f7d9df213cff5a3526f'
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--fetch',action='store_true')
args=p.parse_args()
cutlass=ROOT/'local/cutlass-nvfp4'
if not (cutlass/'include/cute/tensor.hpp').exists():
    if not args.fetch:raise RuntimeError('Run bash demos/pi0/setup.sh first')
    subprocess.run(['git','init',str(cutlass)],check=True)
    subprocess.run(['git','-C',str(cutlass),'fetch','--depth','1','--filter=blob:none',
                    'https://github.com/NVIDIA/cutlass.git',CUTLASS_REV],check=True)
    subprocess.run(['git','-C',str(cutlass),'sparse-checkout','set','include'],check=True)
    subprocess.run(['git','-C',str(cutlass),'checkout',CUTLASS_REV],check=True)
checkout=ROOT/'local/pi0/flash-attention'
if not (checkout/'.git').exists():
    if not args.fetch:raise RuntimeError('Run bash demos/pi0/setup.sh first')
    subprocess.run(['git','init',str(checkout)],check=True)
    subprocess.run(['git','-C',str(checkout),'fetch','--depth','1','--filter=blob:none',
                    'https://github.com/Dao-AILab/flash-attention.git',REV],check=True)
    subprocess.run(['git','-C',str(checkout),'sparse-checkout','set','csrc/flash_attn/src'],check=True)
    subprocess.run(['git','-C',str(checkout),'checkout',REV],check=True)
actual=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
if actual!=REV:raise RuntimeError('Unexpected FlashAttention revision: '+actual)
source=checkout/'csrc/flash_attn/src'
overlay=ROOT/'local/pi0/flash-attention-overlay'
overlay.mkdir(parents=True,exist_ok=True)
records={}
for file in source.iterdir():
    if file.suffix not in ('.h','.hpp','.cuh'):continue
    target=overlay/file.name
    if file.name=='flash_fwd_kernel.h':
        text=file.read_text().replace('__syncthreads();','asm volatile("bar.sync 6, 128;" ::: "memory");')
        text=text.replace('extern __shared__ char smem_[];','char* smem_=params.pi0_smem;',1)
        if target.is_symlink():target.unlink()
        target.write_text(text)
    elif not target.exists():target.symlink_to(file)
    records[file.name]=hashlib.sha256(target.read_bytes()).hexdigest()
(overlay/'manifest.json').write_text(json.dumps({'revision':REV,'headers':records,
    'adaptation':'VM consumer barrier 6/128 and VM-owned physical shared-memory pool'},indent=2)+'\n')
