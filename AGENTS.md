# Scope

- Keep this repository a compact working port of llm.c to Vx, not a tutorial. Explanations belong in a separate Gist or article.
- The CPU port must stay bit-exact with llm.c: run `make test` and `make train-check` after changing `gpt2.vx`, `train.vx` or `data.vx`.
- `train_gpu.vx` is generated: edit `gpu-support/train_gpu.tmpl` and `gpu-support/gen_train_gpu.py`, then run the generator. Run `make gpu-test` and `make gpu-train-check` after GPU changes.
- Rent GPUs only through Fission, quote first, and never repeat a paid step after an unclear outcome.
- Commit measurements in `results/`; keep generated programs and binaries in ignored `build/`.
