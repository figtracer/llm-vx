# The attention backward change on an L4: the small reference and the training
# loop against llm.c, then GPT-2 124M (random weights, T=1024, B=4) with the
# time of each step split into forward, backward and their attention parts.
set -x
OUT=/workspace/out
$VXC ref/test_gpu_small.vx --machine machines/dev.vx -O3 2>/dev/null | grep -v '^\[' > $OUT/test_gpu_small.log
echo "test exit $?"
tail -3 $OUT/test_gpu_small.log
$VXC ref/train_gpu_small.vx --machine machines/dev.vx -O3 2>/dev/null | grep -v '^\[' > $OUT/vx_gpu_train.txt
make train-check >/dev/null 2>&1; python3 ref/compare_train.py build/llmc_train.txt $OUT/vx_gpu_train.txt --tol 0.001 | tee $OUT/compare_small.txt
mkdir -p build/rnd124
cc -O2 ref/make_random.c -o build/make_random && build/make_random build/rnd124 1024 50257 50304 12 12 768 40000
python3 gpu-support/entry.py build/bench_b4.vx 2 4 1024 build/rnd124
$VXC build/bench_b4.vx --machine machines/l4.vx -O3 > $OUT/bench_b4.log 2>&1
echo "B=4 exit $?"
grep -v '^\[flat' $OUT/bench_b4.log | tail -16
