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
| GPU program on an L4 and an H100 | Logits within 6e-8, gradients within 2.1e-8 ([`l4_test_small.log`](../results/gpu/l4_test_small.log), [`h100_test_small.log`](../results/gpu/h100_test_small.log)). |
| GPU training loop on an L4 vs llm.c | 41 training and 5 validation losses equal to every printed digit, same text ([`l4_train_small_vs_llmc.txt`](../results/gpu/l4_train_small_vs_llmc.txt)). |

The CPU port compiles to the same speed as C at `-O3`: one GPT-2 124M step takes 7.8 s on one Mac
core.

## Speed

One training step of GPT-2 124M at B=4, T=1024 on an L4. Every version gives the same five losses
to every digit, and the GPU checks against llm.c pass for each.

| Version | FP32 | `VX_TF32=1` |
|---|---:|---:|
| First version | 24.0 s | |
| O(T) softmax backward; attention backward split by head | 4.4 s | |
| Attention output split by head; gradients zeroed with `cudaMemset` | 1.70 s | |
| Query rows in registers; biases added by cuBLAS | 1.22 s | 1.09 s |
| 64-lane attention kernels; 32-wide elementwise tensors | 0.94 s | 0.81 s |

`VX_TF32=1` runs the matmuls on TF32 tensor cores, as llm.c's `train_gpt2_fp32.cu` does on
Ampere and later; it changes the losses in the fifth or sixth digit.

Where a step goes now ([`l4_b4_phases.log`](../results/gpu/l4_b4_phases.log),
[`l4_kernel_times.txt`](../results/gpu/l4_kernel_times.txt)):

| Part | ms per step |
|---|---:|
| Vx kernels on the GPU | 511 |
| cuBLAS matmuls (FP32) | 219 |
| Everything else, mostly Vx's per-launch overhead (362 launches) | ~210 |

Most of the remaining kernel time is attention: each thread walks its own row of the T×T matrices,
so its accesses do not coalesce, and Vx has no shared memory tiles or vector loads to do better.
Most of the per-launch overhead is Vx's runtime: each launch searches the kernel's payload, which
holds the whole module's PTX (about 374 KB), byte by byte for two fields that are not there. With
Vx rebuilt to search with `memchr` ([Vx#1331](https://github.com/vx-lang/Vx/issues/1331),
`gpu-support/vx-patches/payload-field-memchr.patch`), a step takes 0.84 s instead of 0.95 s, with
the same losses, and the time outside kernels and cuBLAS falls from about 212 ms to 94 ms
([`l4_b4_vx_memchr.log`](../results/gpu/l4_b4_vx_memchr.log)).

Vx's checker counts the peak of live tensors. In these versions a few small buffers that nothing
reads, 0.87 MB at B=4, are no longer live at the peak, so `make admit` prints slightly lower byte
counts than the table above; every verdict is the same.
