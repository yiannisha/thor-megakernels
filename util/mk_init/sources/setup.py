import os
import sys
import subprocess
from pybind11.setup_helpers import build_ext
import pybind11
from setuptools import setup, Extension

# Environment variables
THUNDERKITTENS_ROOT = os.environ.get('THUNDERKITTENS_ROOT', '')
MEGAKERNELS_ROOT = os.environ.get('MEGAKERNELS_ROOT', '')
PYTHON_VERSION = os.environ.get('PYTHON_VERSION', f'{sys.version_info.major}.{sys.version_info.minor}')

# Target GPU (default to HOPPER)
TARGET = os.environ.get('TARGET_GPU', 'HOPPER') # or BLACKWELL

# Source file
SRC = 'src/{{PROJECT_NAME_LOWER}}.cu'

# Get Python include directory
def get_python_include():
    try:
        python_include = subprocess.check_output(['python', '-c', "import sysconfig; print(sysconfig.get_path('include'))"]).decode().strip()
        return python_include
    except subprocess.CalledProcessError:
        return ''

# Base NVCC flags
nvcc_flags = [
    '-DNDEBUG',
    '-Xcompiler=-fPIE',
    '--expt-extended-lambda',
    '--expt-relaxed-constexpr',
    '-Xcompiler=-Wno-psabi',
    '-Xcompiler=-fno-strict-aliasing',
    '--use_fast_math',
    '-forward-unknown-to-host-compiler',
    '-O3',
    '-Xnvlink=--verbose',
    '-Xptxas=--verbose',
    '-Xptxas=--warn-on-spills',
    '-std=c++20',
    '-x', 'cu',
    '-lrt',
    '-lpthread',
    '-ldl',
    '-lcuda',
    '-lcudadevrt',
    '-lcudart_static',
    '-lcublas',
    '-lineinfo',
    '-shared',
    '-fPIC'
]

# Include directories
include_dirs = [
    f'{THUNDERKITTENS_ROOT}/include',
    f'{MEGAKERNELS_ROOT}/include',
    pybind11.get_include(),
    get_python_include()
]

# Get python config flags
def get_python_config_flags():
    try:
        ldflags = subprocess.check_output(['python3-config', '--ldflags']).decode().strip().split()
        return ldflags
    except subprocess.CalledProcessError:
        return []

# Add python config flags
nvcc_flags.extend(get_python_config_flags())

# Conditional setup based on target GPU
if TARGET == 'HOPPER':
    nvcc_flags.extend(['-DKITTENS_HOPPER', '-gencode=arch=compute_90a,code=sm_90a'])
elif TARGET == 'BLACKWELL':
    nvcc_flags.extend(['-DKITTENS_HOPPER', '-DKITTENS_BLACKWELL', '-gencode=arch=compute_100a,code=sm_100a'])
elif TARGET == 'THOR':
    arch = os.environ.get('THOR_ARCH', 'sm_110a')
    if arch not in ('sm_110', 'sm_110a'):
        raise ValueError('THOR_ARCH must be sm_110 or sm_110a')
    nvcc_flags.extend(['-DKITTENS_HOPPER', '-DMK_THOR', f'-gencode=arch={arch.replace("sm_", "compute_")},code={arch}',
                       '-DMK_THOR_STATIC_REGISTERS=' + os.environ.get('THOR_STATIC_REGISTERS', '96'),
                       '-maxrregcount=' + os.environ.get('THOR_STATIC_REGISTERS', '96')])
else:
    raise ValueError(f"Invalid target: {TARGET}")

# Get python extension suffix
def get_extension_suffix():
    try:
        suffix = subprocess.check_output(['python3-config', '--extension-suffix']).decode().strip()
        return suffix
    except subprocess.CalledProcessError:
        return '.so'

# Custom build extension class to use nvcc
class CudaExtension(Extension):
    def __init__(self, name, sources, **kwargs):
        super().__init__(name, sources, **kwargs)

class CudaBuildExt(build_ext):
    def build_extension(self, ext):
        if isinstance(ext, CudaExtension):
            self.build_cuda_extension(ext)
        else:
            super().build_extension(ext)
    
    def build_cuda_extension(self, ext):
        nvcc = os.environ.get('NVCC', 'nvcc')
        
        # Get the output file path
        ext_path = self.get_ext_fullpath(ext.name)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(ext_path), exist_ok=True)
        
        # Build the nvcc command
        cmd = [nvcc] + ext.sources + nvcc_flags + ['-o', ext_path]
        
        # Add include directories
        for include_dir in include_dirs:
            cmd.extend(['-I', include_dir])
        
        print(f"Building CUDA extension with command: {' '.join(cmd)}")
        
        # Execute the command
        subprocess.check_call(cmd)

# Define the extension
ext_modules = [
    CudaExtension(
        '{{PROJECT_NAME_LOWER}}',
        sources=[SRC],
    )
]

setup(
    name='{{PROJECT_NAME_LOWER}}',
    ext_modules=ext_modules,
    cmdclass={'build_ext': CudaBuildExt},
    zip_safe=False,
    python_requires=">=3.6",
)
