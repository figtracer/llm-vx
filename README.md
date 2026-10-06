# llm.vx

GPT-2 training in [Vx](https://github.com/vx-lang/Vx), ported from Karpathy's
[llm.c](https://github.com/karpathy/llm.c). Two programs:

- `train.vx`, `gpt2.vx`: llm.c's CPU training, kernel for kernel.
- `train_gpu.vx`, `gpu.vx`: the same model on the GPU, with every parameter, gradient, AdamW moment and
  activation a tensor in GPU memory. Vx checks at compile time that the whole training step fits a
  given GPU.

## Results

**The CPU port matches llm.c bit for bit.** On a random-weight GPT-2, `make test` compares logits,
loss, all 16 parameter gradients and ten AdamW steps with llm.c's C code; every difference is 0.
`make train-check` runs llm.c's own `train_gpt2` and the Vx loop on the same files: all 41 training
losses, 5 validation losses and 126 sampled tokens are identical, shuffled batch order included.
Vx compiles the kernels with the same speed as C at `-O3` (12 ms vs 13 ms per step, single thread).

**The compiler knows the largest batch before a GPU is rented.** `gpu-support/admit.py` compiles
`train_gpu.vx` for GPT-2 124M at T=1024 against each GPU's capacity, in about 0.2 s per verdict:

| GPU | Largest batch, Vx's fleet capacity | Largest batch, measured allocatable memory |
|---|---:|---:|
| A100-40GB | 8 | 8 |
| A100-80GB | 18 | |
| H100 | 18 | 17 |
| H200 | 32 | 31 |
| B200 | 44 | 40 |

The byte count Vx reports matches llm.c's arithmetic exactly (2,578,399,232 bytes at B=4, T=64).
Measured allocatable memory is from [vx-fit](https://github.com/figtracer/vx-fit).

## Run

Requires Rust, LLVM 22, Z3, CMake and Python 3.10+.

```sh
git clone --recurse-submodules <this repo> llm-vx
cd llm-vx
(cd Vx && ./setup.sh && source config.local && cargo build --release --locked --bin vxc -p vxc)
make test             # CPU port vs llm.c, bit for bit
make train-check      # training loop vs llm.c's train_gpt2
make gpu-test         # GPU program vs llm.c; kernels run on the CPU without CUDA
python3 gpu-support/admit.py 1024
```

GPT-2 124M itself needs llm.c's starter pack in `data/` (`gpt2_124M.bin`,
`gpt2_124M_debug_state.bin`, `gpt2_tokenizer.bin`, `tiny_shakespeare_{train,val}.bin`); then
`make train`. On an NVIDIA GPU, `gpu-support/remote.sh` installs the toolchain and builds Vx with
its CUDA plugin.

## How the GPU program is written

Vx runs a `spawn` loop in parallel only when every write is indexed by the loop variable, so each
tensor is 2-D (rows, columns) in llm.c's memory order and each kernel computes one row per GPU
thread. A kernel that updates one layer covers the whole tensor and skips the other rows. Matrix
multiplications call cuBLAS, as llm.c's fp32 CUDA version does. Attention backward gathers instead
of scattering, because Vx has no atomics.

## Limits and workarounds

- **Machine files.** The program uses the built-in `Topology::GPU` and `Memory::GPU_HBM`: Vx's CUDA
  runtime sends only built-in topologies to a device, so the fleet files' `Topology Device` would
  run on the host. `machines/*.vx` give `GPU_HBM` its capacity, `within: Memory::CPU_DRAM` (kernels
  cannot take host scalars otherwise, Vx#850) and `managed: cached` (Vx refuses non-matmul kernels
  on explicitly managed memory).
- **Shapes.** Tensor extents must be generic parameters: module constants cannot size a tensor, and
  `[R * C]` does not match a parameter `[N]`. `train_gpu.vx` therefore takes 16 shape parameters.
- **C shim.** `gpu-support/shim.c` offsets device pointers for cuBLAS (Vx has no pointer
  arithmetic) and copies batches in and losses out. `CLANG_PATH` links it, since `vxc` has no flag
  for extra objects.
- **Numerics.** The GPU program sums in a different order from llm.c in cuBLAS and writes GELU's
  `sech^2` as `1 - tanh^2`; on a CPU it matches llm.c to 1e-8 in the logits and to every printed
  digit of the training losses.
- **llm.c's reference** strips `#pragma float_control(precise, on)` from `train_gpt2.c`, which
  turns FMA contraction back on in `gelu_backward` and breaks bit-exact comparison.
