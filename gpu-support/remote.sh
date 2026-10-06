#!/bin/bash
# Runs on a rented NVIDIA GPU (a Fission sandbox): installs LLVM 22, Rust and
# the CUDA headers and libraries, builds Vx with its CUDA plugin, then runs the
# commands given as arguments from /workspace/llm-vx with the toolchain set up.
#
# Expects /workspace/llm-vx.tgz (this repository and Vx's sources).
set -euo pipefail
log() { echo "== $(date +%T) $*"; }

log machine
nproc
free -g
df -h /workspace | tail -1
head -2 /etc/os-release
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader

mkdir -p /workspace/out
cd /workspace
tar xzf llm-vx.tgz

export DEBIAN_FRONTEND=noninteractive
log apt
apt-get update -qq
apt-get install -y -qq --no-install-recommends build-essential cmake ninja-build pkg-config git curl \
  wget ca-certificates gnupg lsb-release software-properties-common libffi-dev zlib1g-dev libzstd-dev \
  libedit-dev libxml2-dev z3 libz3-dev python3-pip >/dev/null

log llvm 22
wget -qO /tmp/llvm.sh https://apt.llvm.org/llvm.sh
bash /tmp/llvm.sh 22 >/dev/null
apt-get install -y -qq llvm-22 llvm-22-dev llvm-22-tools clang-22 lld-22 libmlir-22-dev mlir-22-tools \
  libpolly-22-dev >/dev/null
export PATH=/usr/lib/llvm-22/bin:$PATH

# Vx looks for a toolkit with include/cuda_runtime.h and lib64/libcudart.so.
# NVIDIA's wheels have the headers, the runtime, cuBLAS and libdevice; the
# driver library comes with the GPU.
log cuda
pip install -q --break-system-packages nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cuda-nvcc-cu12 2>/dev/null \
  || pip install -q nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cuda-nvcc-cu12
NV=$(python3 -c 'import nvidia; print(list(nvidia.__path__)[0])')
mkdir -p /opt/cuda/include /opt/cuda/lib64/stubs
cp -r "$NV"/cuda_runtime/include/. "$NV"/cublas/include/. /opt/cuda/include/
for lib in "$NV"/cuda_runtime/lib/libcudart.so.12 "$NV"/cublas/lib/libcublas.so.12 "$NV"/cublas/lib/libcublasLt.so.12; do
  base=$(basename "$lib")
  ln -sf "$lib" /opt/cuda/lib64/"$base"
  ln -sf "$lib" /opt/cuda/lib64/"${base%.12}"
done
DRIVER=$(ldconfig -p | awk '/libcuda.so.1 / {print $NF; exit}')
[ -n "$DRIVER" ] || DRIVER=$(find / -name 'libcuda.so.1' 2>/dev/null | head -1)
ln -sf "$DRIVER" /opt/cuda/lib64/stubs/libcuda.so
ls -la /opt/cuda/lib64 /opt/cuda/lib64/stubs
test -f /opt/cuda/include/cuda.h
export CUDA_HOME=/opt/cuda
export VX_LIBDEVICE="$NV"/cuda_nvcc/nvvm/libdevice/libdevice.10.bc
export LD_LIBRARY_PATH=/opt/cuda/lib64:${LD_LIBRARY_PATH:-}
test -f "$VX_LIBDEVICE"

log rust
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null
source "$HOME/.cargo/env"

log build vx
cd /workspace/llm-vx/Vx
./setup.sh >/dev/null
source config.local
cargo build --release --locked --bin vxc -p vxc 2>&1 | tail -3
cargo build --release --locked -p vx_std_core 2>&1 | tail -1

log build shim
cd /workspace/llm-vx
mkdir -p build
cc -O2 -DVX_CUDA -I/opt/cuda/include -c gpu-support/shim.c -o build/shim.o
export VX_STD_PATH=$PWD/Vx/stdlib/std:$PWD/Vx/stdlib
export VX_SHIM=$PWD/build/shim.o
export CLANG_PATH=$PWD/gpu-support/clang-link.sh
export VXC=$PWD/Vx/target/release/vxc

log run
"$@"
log done
