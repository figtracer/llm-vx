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
  53 kernels computes one row per GPU thread. A kernel that updates one layer covers the whole
  tensor and skips the other rows.
- **cuBLAS for matrix multiplies**, as llm.c's fp32 CUDA version does.
- **No atomics in Vx**, so attention backward gathers instead of scattering.
- **Generated entry point.** `train_gpu.vx` is generated from `gpu-support/train_gpu.tmpl` by
  `gpu-support/gen_train_gpu.py`; `gpu-support/entry.py` writes a `main` for one model size and
  batch.
- **Three modes.** 0 checks against llm.c's reference (`ref/test_gpu_small.vx`), 1 trains like
  llm.c (`ref/train_gpu_small.vx`), 2 runs five steps and prints where the time goes.

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
