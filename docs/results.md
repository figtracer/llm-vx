# Results

All runs use Vx at 3d59497a and llm.c at f1e2ace. GPU runs were on rented NVIDIA cards through
[Fission](https://github.com/figtracer/fission) (driver 580.95.05). Logs are in
[`results/`](../results).

## The compiler knows the largest batch

GPT-2 124M at T=1024. Vx counts every parameter, gradient, AdamW moment and activation of a
training step: 1,991,614,464 bytes plus 4,612,063,232 per sequence in the batch. Each program was
compiled against two capacities: the one a machine file declares, and the bytes one process can
allocate on that card (`gpu-support/probe.py`). Then it ran on the card.

| GPU | Batch | Bytes Vx counts | Declared capacity | Measured capacity | On the card |
|---|---:|---:|---|---|---|
| L4 | 4 | 20,439,867,392 | admitted (24 GiB) | admitted (23,452,450,816 B) | trains |
| L4 | 5 | 25,051,930,624 | admitted | **rejected** | `cudaMalloc` out of memory |
| H100 | 17 | 80,396,689,408 | admitted (80 GiB, Vx's fleet file) | admitted (84,462,796,800 B) | trains |
| H100 | 18 | 85,008,752,640 | admitted | **rejected** | `cudaMalloc` out of memory |

B=18 needs 0.65% more than the H100 lets one process allocate. Vx's fleet file declares 80 GiB
(85.9 GB); the card reports 85.5 GB, and a process gets 84.46 GB of it. The byte count also matches
llm.c's own arithmetic (2,578,399,232 bytes at B=4, T=64).

Logs: [`l4_b4.log`](../results/gpu/l4_b4.log), [`l4_b5.log`](../results/gpu/l4_b5.log),
[`h100_b17.log`](../results/gpu/h100_b17.log), [`h100_b18.log`](../results/gpu/h100_b18.log),
[`l4_probe.txt`](../results/gpu/l4_probe.txt), [`h100_probe.txt`](../results/gpu/h100_probe.txt).

`make admit` gives the same answer for other GPUs, in about 0.2 s per verdict
([`admission_t1024.json`](../results/admission_t1024.json)):

| GPU | Largest batch, declared capacity | Largest batch, measured capacity |
|---|---:|---:|
| L4 | 5 | 4 |
| A100-40GB | 8 | 8 |
| A100-80GB | 18 | |
| H100 | 18 | 17 |
| H200 | 32 | 31 |
| B200 | 44 | 40 |

Measured capacities other than the L4's are from [vx-fit](https://github.com/figtracer/vx-fit).
Our H100 could allocate 2.1 MB more than vx-fit's.

## The port is correct

| Check | Result |
|---|---|
| CPU port vs llm.c's C code, random-weight GPT-2 (`make test`) | Logits, loss, all 16 gradients and ten AdamW steps identical: every difference is 0. |
| CPU training loop vs llm.c's `train_gpt2` (`make train-check`) | Every loss and every sampled token identical, shuffled batch order included. |
| GPT-2 124M vs llm.c's PyTorch reference | Logits within 1.4e-3, loss 5.26989 vs 5.27001, every gradient within llm.c's tolerances, ten training losses down to 0.378 vs 0.3765. |
| GPT-2 124M on tiny Shakespeare (`make train`) | The same 41 training losses, 5 validation losses (5.325 to 4.294) and generated text as llm.c ([`results/cpu/`](../results/cpu)). |
| GPU program on an L4 and an H100 | Logits within 6e-8, gradients within 2.1e-8 ([`l4_test_small.log`](../results/gpu/l4_test_small.log), [`h100_test_small.log`](../results/gpu/h100_test_small.log)). The current version, on an L4: logits within 4.5e-8, gradients within 1.9e-8 ([`test_gpu_small.log`](../results/gpu/speed/test_gpu_small.log)). |
| GPU training loop on an L4 vs llm.c | 41 training and 5 validation losses equal to every printed digit, same text ([`l4_train_small_vs_llmc.txt`](../results/gpu/l4_train_small_vs_llmc.txt)). The current version, which sums in cuBLAS's order as llm.c's CUDA trainer does: within 1e-6, same text ([`compare_small.txt`](../results/gpu/speed/compare_small.txt)). |

The CPU port compiles to the same speed as C at `-O3`: one GPT-2 124M step takes 7.8 s on one Mac
core.

## Speed

One training step of GPT-2 124M at B=4, T=1024 on an L4, the mean of steps 1 to 4. Each of the
last three rows ran on its own rented L4, with llm.c's CUDA trainer measured on the same card
(0.44, 0.46 and 0.43 s in FP32; [`results/gpu/speed/`](../results/gpu/speed)).

| Version | FP32 | `VX_TF32=1` |
|---|---:|---:|
| First version | 24.0 s | |
| O(T) softmax backward; attention backward split by head | 4.4 s | |
| Attention output split by head; gradients zeroed with `cudaMemset` | 1.70 s | |
| Query rows in registers; biases added by cuBLAS | 1.22 s | 1.09 s |
| 64-lane attention kernels; 32-wide elementwise tensors | 0.94 s | 0.81 s |
| Attention's products, residual additions and bias gradients on cuBLAS; softmax statistics in parts | 0.62 s | 0.43 s |
| Each attention row's statistics combined once, in one pass | 0.61 s | 0.41 s |
| A fast f32 `exp` and `tanh` in the kernels | 0.53 s | 0.34 s |

`VX_TF32=1` runs the matmuls on TF32 tensor cores, as llm.c's `train_gpt2_fp32.cu` does on
Ampere and later; it changes the losses in the fifth or sixth digit. Up to 0.94 s every version
gives the same five losses to every digit. From the next row on, sums run in cuBLAS's order and
the softmax is computed in parts, as in llm.c's CUDA trainer, so the losses change in the sixth or
seventh digit. Against llm.c on the L4, the GPU program's logits are within 4.5e-8, its gradients
within 1.9e-8 and the small training run's losses within 1e-6
([`test_gpu_small.log`](../results/gpu/speed/test_gpu_small.log),
[`compare_small.txt`](../results/gpu/speed/compare_small.txt)).

**Attention on cuBLAS.** llm.c computes QKᵀ, att·V and their four gradients with
`cublasSgemmStridedBatched`, one product per (batch, head). It first copies q, k and v into a
(B, NH, T, hs) buffer. Here the leading dimension of each operand is the row length of `qkv` or
`atty`, so cuBLAS reads each head where it lies, with one call per batch and no copy. Between the
products, the softmax needs each row's max and sum. A Vx kernel cannot share a reduction among
threads, so one kernel writes 32 partial statistics per row, each over every 32nd key, so that
the threads of a warp read consecutive keys; a second combines them; a third writes the softmax.
The softmax backward needs each row's sum of att · datt, which equals dout · out for the head, so
it costs 64 multiply-adds instead of a pass over the row. cuBLAS also adds the residual
connections (`cublasSgeam`) and sums the bias gradients (`cublasSgemv`), and the backward pass
reads the residual stream's gradient directly instead of copying it. These cut the Vx launches per
step from 362 to 160.

**Narrow tensors.** The attention matrices are declared 4 wide, the fastest of the widths tried
([`widths.txt`](../results/gpu/speed/widths.txt)), and the elementwise ones 2 wide: neighbouring
threads read neighbouring elements, and AdamW's 124M parameters take 15 ms instead of 32.

**A fast exp.** Vx's `f32.exp()` and `tanh()` are LLVM libc's correctly rounded functions, written
in Vx. On a GPU they run in f64, which an L4 does at 1/64 of the f32 rate, with lookup tables in
memory; the exp-heavy kernels were most of the kernel time. The math dialect's `math.exp` lowers to
libdevice's `__nv_expf`, one `ex2.approx`, but a kernel cannot call it: Vx runs a `spawn` loop in
parallel only if it calls nothing but a few stdlib methods, and a call to a wrapper function, or
inline MLIR, makes the kernel run on one thread, with no warning. So `fast_exp!` is a macro: x
clamped to [-87, 88], reduced by a whole number m of ln 2, a degree-6 Taylor polynomial and
`powi` for 2^-m, all allowed. Its error is below 2.5e-7 relative, about what `__expf` gives llm.c,
which builds with `--use_fast_math`. Vx kernel time per step fell from 235 ms to 144 ms.
`gpu-support/serial_kernels.py` lists any kernel that would run on one thread.

Where a step goes now ([`phases_b4.log`](../results/gpu/speed/phases_b4.log),
[`kernel_times_b4.txt`](../results/gpu/speed/kernel_times_b4.txt)):

| Part | ms per step |
|---|---:|
| cuBLAS: dense layers, residual additions, bias gradients | ~250 |
| cuBLAS: attention's products | ~100 |
| Vx kernels (layernorms 58, attention softmax 45, AdamW 15) | 144 |
| Launches and the rest | ~30 |

The layernorm kernels are the largest Vx cost: one thread per row is 4,096 threads, where llm.c
gives each row a warp. Each Vx launch also searches the kernel's payload, which holds the whole
module's PTX, byte by byte for two fields that are not there. With Vx rebuilt to search with
`memchr` ([Vx#1331](https://github.com/vx-lang/Vx/issues/1331),
`gpu-support/vx-patches/payload-field-memchr.patch`), a step takes 20 ms less.

Vx's checker counts the peak of live tensors. A few small buffers that nothing reads, 0.84 MB at
B=4, are not live at the peak, so `make admit` prints slightly lower byte counts than the table
above; every verdict is the same. `d_attproj` and `d_fcproj`, which llm.c's CPU code fills with
copies of the residual gradient, are zeroed each step though nothing reads them, so that the
memory Vx counts stays llm.c's.

## Against llm.c's CUDA trainer

llm.c's `train_gpt2_fp32.cu` and llm.vx on the same L4, with the same random GPT-2 124M weights and
tokens, B=4, T=1024 (`gpu-support/jobs/speed.sh`). llm.c is built with nvcc from NVIDIA's CUDA 12.9
archives. It turns TF32 on by itself on this GPU, so its FP32 build is a copy with that switched
off. Mean step time after the first step ([`results/gpu/speed/`](../results/gpu/speed)):

| | FP32 | TF32 |
|---|---:|---:|
| llm.c | 0.43 s | 0.32 s |
| llm.vx | 0.53 s | 0.34 s |
| llm.vx, with Vx's launch fix ([Vx#1331](https://github.com/vx-lang/Vx/issues/1331)) | 0.51 s | 0.32 s |

Before attention moved to cuBLAS, llm.vx took 0.95 s and 0.81 s
([`vs-llmc/`](../results/gpu/vs-llmc)). With TF32, llm.vx is within 6% of llm.c, and level with
it with the launch fix. In FP32 the gap is 0.09 s, and most of it closes with TF32, so it lies in
the matrix products rather than in the Vx kernels. One candidate, not measured on its own: llm.c
copies q, k and v into contiguous buffers before attention's products, and here cuBLAS reads them
strided through `qkv`.
