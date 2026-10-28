# Compiler
NVCC?=nvcc

ifndef GPU
GPU=B200
endif

TARGET=mk_llama
SRC=llama.cu

# Python version configuration
ifndef PYTHON_VERSION
PYTHON_VERSION=$(shell python3 -c 'import sys; print(str(sys.version_info.major)+"."+str(sys.version_info.minor))')
endif

NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -O3 -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -x cu -lrt -lpthread -ldl -lcuda -lcudadevrt -lcudart_static -lcublas -lineinfo
NVCCFLAGS+= -I${THUNDERKITTENS_ROOT}/include -I${MEGAKERNELS_ROOT}/include $(shell python3 -m pybind11 --includes) $(shell python3-config --ldflags) -shared -fPIC

# Conditional setup based on the target GPU
ifeq ($(GPU),4090)
NVCCFLAGS+= -DKITTENS_4090 -arch=sm_89
else ifeq ($(GPU),A100)
NVCCFLAGS+= -DKITTENS_A100 -arch=sm_80
else ifeq ($(GPU),H100)
NVCCFLAGS+= -DKITTENS_HOPPER -gencode=arch=compute_90a,code=sm_90a
else ifeq ($(GPU),B200)
NVCCFLAGS+= -DKITTENS_HOPPER -DKITTENS_BLACKWELL -gencode=arch=compute_100a,code=sm_100a
else ifeq ($(GPU),THOR)
# TMA/SIMT compatibility adapter; no B200 tensor-memory paths.
THOR_WORKERS?=20
THOR_BUILD_HASH=$(shell PYTHONPATH="$(MEGAKERNELS_ROOT)" python3 -c 'from tools.thor.repro import source_hash; print(source_hash())')
NVCCFLAGS+= -DMK_BUILD_SOURCE_HASH='"$(THOR_BUILD_HASH)"'
NVCCFLAGS+= -DKITTENS_HOPPER -DMK_THOR -DMK_ACTIVE_WORKERS=$(THOR_WORKERS) -gencode=arch=compute_110a,code=sm_110a -maxrregcount=96
else
$(error Unsupported GPU '$(GPU)')
endif

NVCCFLAGS+= $(EXTRA_NVCCFLAGS)

NVCCFLAGS := $(NVCCFLAGS)
export NVCCFLAGS

# Default target
all: $(TARGET)

# run: $(TARGET)
# 	python test.py

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)$(shell python3-config --extension-suffix)
	python3 ../../tools/thor/build_manifest.py --binary "$(TARGET)$(shell python3-config --extension-suffix)" --compiler "$(NVCC)" --gpu "$(GPU)"

# Clean target
clean:
	rm -f $(TARGET)$(shell python3-config --extension-suffix)

.PHONY: dry-run capability-check
dry-run:
	@echo $(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)$(shell python3-config --extension-suffix)
capability-check:
	python3 ../../tools/thor/inspect_device.py
