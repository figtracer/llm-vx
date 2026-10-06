<div align="center">

# llm.vx

**Karpathy's [llm.c](https://github.com/karpathy/llm.c), rewritten in [Vx](https://github.com/vx-lang/Vx).**

[Results](docs/results.md) | [GPU program](docs/gpu.md)

</div>

---

## Highlights

- ✅ Matches llm.c bit for bit on the CPU.
- 🖥️ Matches llm.c on an L4 and an H100.
- 🎯 The compiler knows if a training step fits in GPU memory, exactly.

## Quick start

```sh
git clone --recurse-submodules https://github.com/figtracer/llm-vx.git
cd llm-vx
(cd Vx && ./setup.sh && source config.local && cargo build --release --locked --bin vxc -p vxc)
make test
```

## License

[MIT](LICENSE). Based on [llm.c](https://github.com/karpathy/llm.c) by Andrej Karpathy.
