<div align="center">

# llm.vx

**Karpathy's [llm.c](https://github.com/karpathy/llm.c), rewritten in [Vx](https://github.com/vx-lang/Vx).**

[Results](docs/results.md) | [GPU program](docs/gpu.md)

</div>

---

GPT-2 training, kernel for kernel, on the CPU and on NVIDIA GPUs. In Vx a tensor's type says where
it lives, so the compiler can also check that a whole training step fits in GPU memory.

## Highlights

- Matches llm.c bit for bit on the CPU: every logit, gradient and loss.
- Matches llm.c on NVIDIA GPUs: losses within a millionth, gradients within 2e-8.
- Within 6% of llm.c's own CUDA trainer on an L4 (TF32), with attention on cuBLAS as llm.c does.
- The compiler's memory check is exact: on an H100 it admits batch 17 and rejects 18, and the card
  agrees.

## Quick start

```sh
git clone --recurse-submodules https://github.com/figtracer/llm-vx.git
cd llm-vx
(cd Vx && ./setup.sh && source config.local && cargo build --release --locked --bin vxc -p vxc)
make test        # CPU port vs llm.c
make gpu-test    # GPU program vs llm.c (runs on the CPU without CUDA)
make admit       # largest batch per GPU, from the compiler
```

## License

[MIT](LICENSE). Based on [llm.c](https://github.com/karpathy/llm.c) by Andrej Karpathy.
