# The L4 run: the small reference and training loop on the real GPU against
# llm.c, the card's allocatable memory, and GPT-2 124M (random weights, T=1024)
# at B=4 and B=5. A 24 GiB machine file admits both; B=5 needs 25.05 GB.
set -x
OUT=/workspace/out
VX_DISPATCH_VERBOSE=1 $VXC ref/test_gpu_small.vx --machine machines/dev.vx -O3 > $OUT/test_gpu_small.log 2>&1
echo "test exit $?"
grep -v '^\[' $OUT/test_gpu_small.log | grep -iv 'dispatch\|launch' | tail -32
grep -ic 'launch' $OUT/test_gpu_small.log
grep -i 'launch\|cublas\|dispatch' $OUT/test_gpu_small.log | sort | uniq -c | sort -rn | head -12
$VXC ref/train_gpu_small.vx --machine machines/dev.vx -O3 2> $OUT/train_gpu_small.err | grep -v '^\[' > $OUT/vx_gpu_train.txt
echo "train exit $?"
make train-check >/dev/null 2>&1; python3 ref/compare_train.py build/llmc_train.txt $OUT/vx_gpu_train.txt --tol 0.001 | tee $OUT/compare_small.txt
python3 gpu-support/probe.py | tee $OUT/probe.txt
mkdir -p build/rnd124
cc -O2 ref/make_random.c -o build/make_random && build/make_random build/rnd124 1024 50257 50304 12 12 768 40000
for B in 4 5; do
  python3 gpu-support/entry.py build/bench_b$B.vx 2 $B 1024 build/rnd124
  $VXC build/bench_b$B.vx --machine machines/l4.vx -O3 > $OUT/bench_b$B.log 2>&1
  echo "B=$B exit $?"
  grep -v '^\[flat' $OUT/bench_b$B.log | tail -14
done
ls -la $OUT
