# Vx toolchain built in the Vx submodule (see README).
VXC := Vx/target/release/vxc
VXENV := source Vx/config.local && export VX_STD_PATH=$(CURDIR)/Vx/stdlib/std:$(CURDIR)/Vx/stdlib &&
SHELL := /bin/bash

.PHONY: test ref train train-check clean

test: build/ref_model.bin
	$(VXENV) $(VXC) test_gpt2.vx -O3

ref: build/ref_model.bin

# llm.c wraps gelu_backward in `#pragma float_control(precise, on)`, which turns
# FMA contraction back on. Strip it so the reference does plain f32 arithmetic,
# as Vx does.
build/ref_model.bin: ref/make_ref.c
	mkdir -p build
	sed '/#pragma float_control/d' llm.c/train_gpt2.c > build/train_gpt2.c
	$(CC) -O3 -ffp-contract=off -Wno-unused-result -Wno-gnu-folding-constant -I build -I llm.c ref/make_ref.c -lm -o build/make_ref
	./build/make_ref build

# Trains GPT-2 124M on tiny Shakespeare. Needs the starter pack in data/.
train:
	$(VXENV) $(VXC) train_gpt2.vx -O3

# Runs llm.c's own train_gpt2 and the Vx training loop on the small reference
# model and data, and compares every loss and every sampled token.
train-check: build/ref_model.bin
	$(CC) -O3 -ffp-contract=off -Wno-unused-result -Wno-gnu-folding-constant -I build -I llm.c build/train_gpt2.c -lm -o build/llmc_train
	rm -rf build/llmc_run && mkdir -p build/llmc_run/dev/data/tinyshakespeare
	ln -s ../ref_model.bin build/llmc_run/gpt2_124M.bin
	ln -s ../tokenizer.bin build/llmc_run/gpt2_tokenizer.bin
	ln -s ../../../../train_tokens.bin build/llmc_run/dev/data/tinyshakespeare/tiny_shakespeare_train.bin
	ln -s ../../../../val_tokens.bin build/llmc_run/dev/data/tinyshakespeare/tiny_shakespeare_val.bin
	cd build/llmc_run && ../llmc_train > ../llmc_train.txt
	$(VXENV) $(VXC) ref/train_small.vx -O3 2>/dev/null | grep -v '^\[' > build/vx_train.txt
	python3 ref/compare_train.py build/llmc_train.txt build/vx_train.txt

clean:
	rm -rf build
