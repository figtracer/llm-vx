#!/bin/bash
# Runs on a rented NVIDIA GPU (a Fission sandbox): installs LLVM 22, Rust and
# the CUDA headers and libraries, builds Vx with its CUDA plugin, then runs the
# commands given as arguments from /workspace/llm-vx with the toolchain set up.
#
# Expects /workspace/llm-vx.tgz (or $VX_BUNDLE): this repository without its submodules, plus
# llm-vx/submodules.txt naming the Vx and llm.c commits to fetch. Uploads go
# through Fission 4 KiB at a time, so the sandbox fetches large sources itself.
set -euo pipefail
# Fission sandboxes run with a private umask, so a key written to /etc/apt is
# unreadable to apt's own user and the LLVM repository counts as unsigned.
umask 022
log() { echo "== $(date +%T) $*"; }

# The facts below are informational: none may stop the run (Modal's image has
# no `free`, for one).
log machine
nproc || true
grep MemTotal /proc/meminfo || true
df -h /workspace | tail -1 || true
head -2 /etc/os-release || true
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader || true

mkdir -p /workspace/out
cd /workspace
tar xzf "${VX_BUNDLE:-llm-vx.tgz}"

export DEBIAN_FRONTEND=noninteractive
log apt
# A rerun on the same sandbox finds the LLVM repository already added.
chmod a+r /etc/apt/trusted.gpg.d/*.asc 2>/dev/null || true
apt-get update -qq
apt-get install -y -qq --no-install-recommends build-essential cmake ninja-build pkg-config git curl \
  wget ca-certificates gnupg lsb-release libffi-dev zlib1g-dev libzstd-dev \
  libedit-dev libxml2-dev z3 libz3-dev python3-pip >/dev/null
# LLVM's installer needs add-apt-repository before Debian 13 only; the package
# that provides it is gone from Debian 13, which Modal's image runs.
apt-get install -y -qq --no-install-recommends software-properties-common >/dev/null 2>&1 || true

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
pip install -q --break-system-packages nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cuda-nvcc-cu12 nvidia-cuda-cccl-cu12 2>/dev/null \
  || pip install -q nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cuda-nvcc-cu12 nvidia-cuda-cccl-cu12
NV=$(python3 -c 'import nvidia; print(list(nvidia.__path__)[0])')
mkdir -p /opt/cuda/include /opt/cuda/lib64/stubs
# The runtime headers include crt/ (from nvcc) and nv/target (from CCCL).
cp -r "$NV"/cuda_runtime/include/. "$NV"/cublas/include/. "$NV"/cuda_nvcc/include/. "$NV"/cuda_cccl/include/. \
  /opt/cuda/include/
for lib in "$NV"/cuda_runtime/lib/libcudart.so.12 "$NV"/cublas/lib/libcublas.so.12 "$NV"/cublas/lib/libcublasLt.so.12; do
  base=$(basename "$lib")
  ln -sf "$lib" /opt/cuda/lib64/"$base"
  ln -sf "$lib" /opt/cuda/lib64/"${base%.12}"
done
DRIVER=$(ldconfig -p | awk '/libcuda.so.1 / {print $NF; exit}')
[ -n "$DRIVER" ] || DRIVER=$(find / -name 'libcuda.so.1' 2>/dev/null | head -1)
test -f /opt/cuda/include/cuda.h
if [ -n "$DRIVER" ]; then
  ln -sf "$DRIVER" /opt/cuda/lib64/stubs/libcuda.so
  export CUDA_HOME=/opt/cuda
  SHIM_CUDA=-DVX_CUDA
else
  # No NVIDIA driver: build Vx without its CUDA plugin, so the same run works
  # on any Linux host, with kernels on the CPU.
  log "no NVIDIA driver; building Vx without CUDA"
  export VX_DISABLE_CUDA=1
  SHIM_CUDA=
fi
export VX_LIBDEVICE="$NV"/cuda_nvcc/nvvm/libdevice/libdevice.10.bc
export LD_LIBRARY_PATH=/opt/cuda/lib64:${LD_LIBRARY_PATH:-}
test -f "$VX_LIBDEVICE"

log rust
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null
source "$HOME/.cargo/env"

log sources
cd /workspace/llm-vx
if [ -f submodules.txt ]; then
  while read -r dir url sha; do
    rm -rf "$dir"
    git init -q "$dir"
    git -C "$dir" fetch -q --depth 1 "$url" "$sha"
    git -C "$dir" checkout -q FETCH_HEAD
  done < submodules.txt
fi
[ -f build/ref_model.bin ] || make ref >/dev/null

log build vx
cd /workspace/llm-vx/Vx
./setup.sh >/dev/null
source config.local
# Sandboxes have one vCPU, and the release profile's thin LTO with one codegen
# unit is the slowest build there is. vxc's own speed hardly matters here: the
# programs it emits are optimized by LLVM at -O3 either way.
export CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 CARGO_PROFILE_RELEASE_OPT_LEVEL=1
build() {
  if ! cargo build --release --locked "$@" > /workspace/out/cargo.log 2>&1; then
    tail -60 /workspace/out/cargo.log
    exit 1
  fi
  tail -1 /workspace/out/cargo.log
}
build --bin vxc -p vxc
build -p vx_std_core

log build shim
cd /workspace/llm-vx
mkdir -p build
cc -O2 $SHIM_CUDA -I/opt/cuda/include -c gpu-support/shim.c -o build/shim.o
export VX_STD_PATH=$PWD/Vx/stdlib/std:$PWD/Vx/stdlib
export VX_SHIM=$PWD/build/shim.o
export CLANG_PATH=$PWD/gpu-support/clang-link.sh
export VXC=$PWD/Vx/target/release/vxc

log run
"$@"
log done
