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
  the loop variable. So each tensor is 2-D (rows, columns) in llm.c's memory order, and each of the
  41 kernels computes one row per GPU thread. A kernel that updates one layer covers the whole
  tensor and skips the other rows.
- **32-wide tensors.** `fch`, `fch_gelu`, `logits`, `probs` and their gradients are declared with
  32 columns instead of a row's width: the same bytes in the same order, so cuBLAS does not notice,
  and the elementwise kernels on them get 32 elements per thread instead of a whole row.
- **Attention.** One thread per (batch, head, position), with the head's 64 query, dout or output
  values held in registers across a row of the attention matrix. Vx has no local arrays, so
  `gpu-support/gen_attention.py` writes these kernels to `attention.vx`, unrolled. Where a kernel
  needs a row per position, it writes per head into a buffer that is dead at that point
  (`preatt`, `datt`) and a second kernel gathers the heads.
- **cuBLAS for matrix multiplies**, as llm.c's fp32 CUDA version does.
- **No atomics in Vx**, so attention backward gathers instead of scattering.
- **Unused buffers as scratch.** llm.c's layout has gradients that nothing reads here. `d_losses`
  holds the column of ones that bias addition multiplies by, and `d_lnf_mean`, `d_lnf_rstd` hold
  each row's softmax max and sum. The memory Vx counts stays llm.c's.
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
| A module `const` or `[R * C]` cannot be a tensor extent ([Vx#1310](https://github.com/vx-lang/Vx/issues/1310)). | 16 shape parameters, written by `gen_train_gpu.py`. |
| No pointer arithmetic, and no `vxc` flag to link an extra object. | `gpu-support/shim.c` offsets device pointers and copies batches; `CLANG_PATH` links it. |
| `vx_std_core` does not build on aarch64 Linux ([Vx#1311](https://github.com/vx-lang/Vx/issues/1311)). | Build on x86_64. |

## Numerics

The GPU program sums in a different order from llm.c (cuBLAS, and softmax backward in O(T)) and
writes GELU's `sech^2` as `1 - tanh^2`. On a CPU and on a GPU it matches llm.c within 6e-8 in the logits and to every
printed digit of the training losses.

The CPU port is bit-exact because the reference build strips
`#pragma float_control(precise, on)` from llm.c's `train_gpt2.c`: that pragma turns FMA
contraction back on in `gelu_backward`.
