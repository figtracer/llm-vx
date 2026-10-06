# llm.vx

GPT-2 training in [Vx](https://github.com/vx-lang/Vx), ported from Karpathy's
[llm.c](https://github.com/karpathy/llm.c). Two programs:

- `train.vx`, `gpt2.vx`: llm.c's CPU training, kernel for kernel.
- `train_gpu.vx`, `gpu.vx`: the same model on the GPU, with every parameter, gradient, AdamW moment and
  activation a tensor in GPU memory. Vx checks at compile time that the whole training step fits a
  given GPU.

## Results

**On real GPUs, the compile-time verdict is the hardware's.** For GPT-2 124M at T=1024, Vx counts
every parameter, gradient, AdamW moment and activation of a training step: 20,439,867,392 bytes at
B=4, plus 4,612,063,232 per batch row. We compiled `train_gpu.vx` against two capacities per
card. The first is the one a machine file declares. The second is the number of bytes one process
can allocate on that card, measured by `gpu-support/probe.py`. Then we ran it on rented GPUs:

| GPU | Batch | Bytes Vx counts | Declared capacity | Measured capacity | On the card |
|---|---:|---:|---|---|---|
| L4 | 4 | 20,439,867,392 | admitted (24 GiB) | admitted (23,452,450,816 B) | trains |
| L4 | 5 | 25,051,930,624 | admitted | **rejected** | `cudaMalloc` out of memory |
| H100 | 17 | 80,396,689,408 | admitted (80 GiB, Vx's fleet file) | admitted (84,462,796,800 B) | trains |
| H100 | 18 | 85,008,752,640 | admitted | **rejected** | `cudaMalloc` out of memory |

B=18 needs 0.65% more memory than the H100 lets a process allocate. Vx's fleet file declares 80 GiB
(85.9 GB). The card has 85.5 GB, and a process can get 84.46 GB of it. With the measured capacity,
Vx's verdict matches the hardware on both sides of the boundary, on both cards. Logs are in
`results/l4/` and `results/h100/`.

**The GPU program matches llm.c on a real GPU.** On an NVIDIA L4, and again on the H100,
`ref/test_gpu_small.vx` matches llm.c's C code on a random-weight GPT-2. The logits are within
6e-8, the loss is the same, all 16 parameter gradients are within 2.1e-8, and ten AdamW steps
match. On the L4, the GPU training loop gives llm.c's 41 training losses and 5 validation losses to every
printed digit, and the same generated text. All 52 Vx kernels ran on the GPU, from PTX that Vx
emitted, as parallel launches (`results/l4/kernels.txt`). Matrix multiplications call cuBLAS.

**GPT-2 124M trains on tiny Shakespeare in Vx exactly as in llm.c.** `make train` and llm.c's own
`train_gpt2` produce the same 41 training losses, the same 5 validation losses (5.325 to 4.294) and
the same generated text (`results/cpu/`). Against llm.c's PyTorch reference, `make test` passes:
logits within 1.4e-3, loss 5.26989 vs 5.27001, every gradient within llm.c's tolerances, and the
ten training losses down to 0.378 vs 0.3765. One step takes 7.8 s on one Mac core.

**The CPU port matches llm.c bit for bit.** On a random-weight GPT-2, `make test` compares logits,
loss, all 16 parameter gradients and ten AdamW steps with llm.c's C code; every difference is 0.
`make train-check` does the same for the whole training loop, shuffled batch order included. Vx
compiles the kernels to the same speed as C at `-O3`.

**The compiler knows the largest batch before a GPU is rented.** `gpu-support/admit.py` compiles
`train_gpu.vx` for GPT-2 124M at T=1024 against each GPU's capacity, in about 0.2 s per verdict:

| GPU | Largest batch, declared capacity | Largest batch, measured allocatable memory |
|---|---:|---:|
| L4 | 5 | 4 |
| A100-40GB | 8 | 8 |
| A100-80GB | 18 | |
| H100 | 18 | 17 |
| H200 | 32 | 31 |
| B200 | 44 | 40 |

The byte count Vx reports matches llm.c's arithmetic exactly (2,578,399,232 bytes at B=4, T=64).
Measured allocatable memory is from [vx-fit](https://github.com/figtracer/vx-fit), and for the L4
from the run above. The H100 in our run could allocate 84,462,796,800 bytes, which is 2.1 MB more
than vx-fit's H100.

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
its CUDA plugin. `gpu-support/rent.py L4 l4` rents a GPU through
[Fission](https://github.com/figtracer/fission), runs `gpu-support/jobs/l4.sh` there and downloads
the logs. Without `--approve` it only prints the
quote.

## How the GPU program is written

Vx runs a `spawn` loop in parallel only when every write is indexed by the loop variable, so each
tensor is 2-D (rows, columns) in llm.c's memory order and each kernel computes one row per GPU
thread. A kernel that updates one layer covers the whole tensor and skips the other rows. Matrix
multiplications call cuBLAS, as llm.c's fp32 CUDA version does. Attention backward gathers instead
of scattering, because Vx has no atomics.

## Limits and workarounds

- **Speed.** The GPU program is correct, but not yet fast. The runs above took 24 s per step on both
  cards: B=4 on the L4 and B=17 on the H100. By count, most of that was one loop from llm.c's CPU
  code: softmax backward summed an O(T^2) expression for every row, about T^3/3 multiply-adds per
  attention head, or 200 billion per step. It now uses the O(T) form from llm.c's CUDA version, and dquery, dkey and dvalue
  run one thread per head and position. A run on the GPU with this change is still to come.
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
