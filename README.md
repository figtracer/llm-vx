<div align="center">

# llm.vx

**GPT-2 training in [Vx](https://github.com/vx-lang/Vx), ported from Karpathy's [llm.c](https://github.com/karpathy/llm.c).**

[Quick start](#quick-start) | [Results](docs/results.md) | [GPU program](docs/gpu.md) | [Agent guide](AGENTS.md)

</div>

---

llm.vx trains GPT-2 exactly as llm.c does, written in Vx. Before you rent a GPU, the compiler
tells you the largest batch that fits on it.

## Highlights

- 🎯 On an H100, Vx admits batch 17 and rejects batch 18 at compile time. On the card, 17 trains
  and 18 runs out of memory. Same on an L4 with 4 and 5 ([results](docs/results.md)).
- ✅ The CPU port matches llm.c bit for bit: every logit, gradient and loss.
- 🖥️ The GPU program matches llm.c to every printed digit of the losses, on an L4 and an H100.
- ⚠️ The answer is only as good as the machine file: Vx's own H100 file says 80 GiB and admits 18.

## Quick start

```sh
git clone --recurse-submodules https://github.com/figtracer/llm-vx.git
cd llm-vx
(cd Vx && ./setup.sh && source config.local && cargo build --release --locked --bin vxc -p vxc)
make test    # the CPU port against llm.c, bit for bit
make admit   # the largest GPT-2 124M batch for each GPU, from the compiler alone
```

Requires Rust, LLVM 22, Z3, CMake and Python 3.10+. `make admit` prints, per GPU, the declared
capacity, the largest batch it admits, the measured capacity and the largest batch for that:

```text
H100 80 GiB 18 84460699648 17
```

## Commands

| Command | What it does |
|---|---|
| `make test` | CPU port vs llm.c's C code on a random-weight GPT-2 |
| `make train-check` | CPU training loop vs llm.c's `train_gpt2`, every loss and sampled token |
| `make gpu-test` | GPU program vs llm.c (kernels run on the CPU without CUDA) |
| `make gpu-train-check` | GPU training loop vs llm.c's `train_gpt2` |
| `make admit` | Compile-time batch limits per GPU |
| `make train` | GPT-2 124M on tiny Shakespeare; needs llm.c's starter pack in `data/` |
| `python3 gpu-support/rent.py L4 l4` | Rent a GPU and run a job there ([details](docs/gpu.md#running-on-a-rented-gpu)) |

## What is and is not checked

| | |
| --- | :---: |
| Every byte of a training step is counted at compile time | ✅ |
| The count matches the real card's limit, given its measured capacity | ✅ L4, H100 |
| CPU results identical to llm.c | ✅ |
| GPU results equal to llm.c within float rounding | ✅ |
| Fast | ❌ 4.4 s per step on an L4 at B=4 ([speed](docs/results.md#speed)) |
| Memory the CUDA runtime and cuBLAS use outside the program's tensors | ❌ covered only by measuring capacity |

## Layout

`gpt2.vx`, `train.vx`, `data.vx`: the CPU port. `gpu.vx`, `train_gpu.vx`: the GPU program.
`ref/`: llm.c reference builds and comparisons. `machines/`: machine files. `gpu-support/`:
generator, capacity search, GPU rental. `results/`: logs behind [docs/results.md](docs/results.md).

## License

[MIT](LICENSE). Based on [llm.c](https://github.com/karpathy/llm.c) by Andrej Karpathy. Vx issues
found along the way: [#1308](https://github.com/vx-lang/Vx/issues/1308),
[#1309](https://github.com/vx-lang/Vx/issues/1309), [#1310](https://github.com/vx-lang/Vx/issues/1310),
[#1311](https://github.com/vx-lang/Vx/issues/1311).
