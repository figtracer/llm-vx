# The tight case, on an H100: GPT-2 124M (random weights, T=1024) at B=17 and
# B=18. Vx's fleet file (80 GiB) admits both; B=18 needs 85.01 GB, which is
# 0.65% more than one process could allocate on the H100 vx-fit measured.
set -x
OUT=/workspace/out
$VXC ref/test_gpu_small.vx --machine machines/dev.vx -O3 2>/dev/null | grep -v '^\[' > $OUT/test_gpu_small.log
echo "test exit $?"
tail -3 $OUT/test_gpu_small.log
python3 gpu-support/probe.py | tee $OUT/probe.txt
mkdir -p build/rnd124
cc -O2 ref/make_random.c -o build/make_random && build/make_random build/rnd124 1024 50257 50304 12 12 768 40000
for B in 17 18; do
  python3 gpu-support/entry.py build/bench_b$B.vx 2 $B 1024 build/rnd124
  $VXC build/bench_b$B.vx --machine machines/h100.vx -O3 > $OUT/bench_b$B.log 2>&1
  echo "B=$B exit $?"
  grep -v '^\[flat' $OUT/bench_b$B.log | tail -14
done
ls -la $OUT
