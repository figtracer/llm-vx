# The GPU program

`train_gpu.vx` and `gpu.vx` run GPT-2 training on an NVIDIA GPU. Every parameter, gradient, AdamW
moment and activation is a tensor in `Memory::GPU_HBM`, and all 112 are allocated in one function,
so Vx's capacity check sees a whole training step:

```vx
let qkvw_h = Tensor<f32, [LC3, C]>::uninit();
let mut qkvw = transfer(qkvw_h, Memory::GPU_HBM);
```

Compile it against a machine file that gives `GPU_HBM` a capacity, and the compiler either builds
it or refuses it with `E6010`.

## How it is written

- **One thread per row.** Vx runs a `spawn` loop in parallel only when every write's first index is
  the loop variable, and the loop calls nothing but a few stdlib math methods. So each tensor is
  2-D (rows, columns) in llm.c's memory order, and each of the 20 kernels computes one row per GPU
  thread. A kernel that updates one layer covers the whole tensor and skips the other rows.
  Otherwise a kernel runs on one thread, with no warning; `gpu-support/serial_kernels.py` lists any
  that would.
- **Narrow tensors.** Vx launches up to 4096 blocks of 128 threads, and the threads of a warp take
  consecutive rows. A tensor that only cuBLAS and elementwise kernels write is declared with a few
  columns instead of a row's width: the same bytes in the same order, so cuBLAS does not notice,
  and neighbouring threads read neighbouring elements. The attention matrices are `AW` wide, and
  `fch`, `fch_gelu`, `logits`, `probs`, their gradients, and the large weights and their AdamW
  moments are `EW` wide (4 and 2 by default; `entry.py --aw --ew`).
- **cuBLAS for every matrix product**, as llm.c's fp32 CUDA version does: the dense layers, and
  attention's QKᵀ, att·V and their gradients as strided batched GEMMs, one call per batch for all
  heads. llm.c first copies q, k and v into a (B, NH, T, hs) buffer; here the leading dimensions
  reach each head where it lies in `qkv` and `atty`, so nothing is copied. cuBLAS also adds the
  biases and the residual connections and sums the bias gradients, which saves a Vx launch each.
- **Softmax in parts.** A row's max and sum need the whole row, and Vx has no way for threads to
  share a reduction. So one kernel computes partial statistics in one pass (the online softmax),
  each thread over every 32nd element of a row so that a warp reads consecutive elements; a second
  kernel combines each row's partials; a third writes the softmax. The vocabulary softmax works the
  same way, and the loss and its gradient come from the logits and those statistics, so training
  never writes `probs`.
- **A fast exp.** Vx's `f32.exp()` and `tanh()` are correctly rounded and, on a GPU, run in f64 with
  lookup tables. `fast_exp!` and `fast_tanh!` compute them in f32 to within 2.5e-7, about what
  CUDA's `__expf` gives llm.c, from arithmetic and `round` and `powi`. They are macros because a
  call to a function of our own would make the kernel serial.
- **Softmax backward without a pass over the row.** The sum over keys of att · datt equals
  dout · out for the head (FlashAttention's trick), 64 multiply-adds per row.
- **No copies of the residual gradient.** llm.c's CPU code copies the residual stream's gradient
  into `d_attproj` and `d_fcproj`; here cuBLAS reads `d_residual2` and `d_residual3` directly, and
  adds the residual gradient to the layernorm's. Activation gradients are set, not accumulated, so
  only parameter gradients are zeroed each step.
- **No atomics in Vx**, so the encoder backward gathers instead of scattering.
- **Unused buffers as scratch.** llm.c's layout has buffers that nothing reads at some point of the
  step. `d_losses` holds the column of ones that bias additions multiply by, `d_lnf_mean` and
  `d_lnf_rstd` hold each row's softmax max and sum, `d_att` and `d_logits` hold partial softmax
  statistics in the forward pass, and `preatt` holds each attention row's dot in the backward pass.
  `d_attproj` and `d_fcproj` are zeroed each step though nothing reads them, which keeps them live,
  so the memory Vx counts stays llm.c's.
- **Generated entry point.** `train_gpu.vx` is generated from `gpu-support/train_gpu.tmpl` by
  `gpu-support/gen_train_gpu.py`; `gpu-support/entry.py` writes a `main` for one model size and
  batch.
- **Four modes.** 0 checks against llm.c's reference (`ref/test_gpu_small.vx`), 1 trains like
  llm.c (`ref/train_gpu_small.vx`), 2 times five steps, and 3 also prints where the time goes.
  `gpu-support/kernel_times.py` splits a `VX_TIME_KERNEL=1` run's device time by Vx function.

Without the CUDA plugin (any Mac), Vx runs the kernels on the CPU, so `make gpu-test` and
`make gpu-train-check` work anywhere.

## Running on a rented GPU

`gpu-support/rent.py` rents a GPU through [Fission](https://github.com/figtracer/fission), uploads
this repository, runs a job from `gpu-support/jobs/` and downloads the logs to `build/runs/`:

```sh
python3 gpu-support/rent.py L4 l4                  # print the quote only
python3 gpu-support/rent.py L4 l4 --approve --keep # pay, run, keep the sandbox if the job fails
```

On the sandbox, `gpu-support/remote.sh` installs LLVM 22, Rust and CUDA (from NVIDIA's pip wheels),
fetches the pinned Vx and llm.c commits, and builds Vx with its CUDA plugin. `--reuse NAME --tag T`
runs a job again on a kept sandbox.

## Vx limits and workarounds

| Limit | Workaround here |
|---|---|
| A machine-file `Topology` with `arch: nvptx64` runs its kernels on the GPU but allocates its memory on the host ([Vx#1308](https://github.com/vx-lang/Vx/issues/1308)). | Built-in `Topology::GPU` and `Memory::GPU_HBM`, with the capacity in `machines/*.vx`. |
| A kernel cannot take host scalars or tensor references ([Vx#850](https://github.com/vx-lang/Vx/issues/850)). | `GPU_HBM` is declared `within: Memory::CPU_DRAM`. |
| Vx refuses non-matmul kernels on explicitly managed memory. | `GPU_HBM` is declared `managed: cached`. |
| A write at an offset from the loop variable runs on one thread ([Vx#1309](https://github.com/vx-lang/Vx/issues/1309)). | Loop over the whole tensor and skip rows. |
| A module `const` or `[R * C]` cannot be a tensor extent ([Vx#1310](https://github.com/vx-lang/Vx/issues/1310)). | 25 shape parameters, written by `entry.py`. |
| No pointer arithmetic, and no `vxc` flag to link an extra object. | `gpu-support/shim.c` offsets device pointers and copies batches; `CLANG_PATH` links it. |
| `vx_std_core` does not build on aarch64 Linux ([Vx#1311](https://github.com/vx-lang/Vx/issues/1311)). | Build on x86_64. |
| f32 `exp` and `tanh` run in f64 on a GPU, and a kernel that calls `math.exp` through a function or inline MLIR runs on one thread ([Vx#1376](https://github.com/vx-lang/Vx/issues/1376)). | `fast_exp!` and `fast_tanh!` macros in `gpu.vx`. |

## Numerics

The GPU program sums in a different order from llm.c's CPU code (cuBLAS, softmax statistics in
parts, softmax backward in O(T)), computes `exp` and `tanh` to within 2.5e-7 rather than correctly
rounded, and writes GELU's `sech^2` as `1 - tanh^2`. It matches llm.c's reference within 4.5e-8 in
the logits and 1.9e-8 in the gradients, and its training losses within 1e-6.

The CPU port is bit-exact because the reference build strips
`#pragma float_control(precise, on)` from llm.c's `train_gpt2.c`: that pragma turns FMA
contraction back on in `gelu_backward`.
