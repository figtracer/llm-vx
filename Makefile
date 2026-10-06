# Vx toolchain built in the Vx submodule (see README).
VXC := Vx/target/release/vxc
VXENV := source Vx/config.local && export VX_STD_PATH=$(CURDIR)/Vx/stdlib/std:$(CURDIR)/Vx/stdlib &&
SHELL := /bin/bash

.PHONY: test ref clean

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

clean:
	rm -rf build
