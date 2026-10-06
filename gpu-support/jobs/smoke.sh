# The small reference only: the GPU program's test and training modes against
# llm.c. Quick enough to check a new host before longer runs.
set -x
OUT=/workspace/out
VX_DISPATCH_VERBOSE=1 $VXC ref/test_gpu_small.vx --machine machines/dev.vx -O3 > $OUT/test_gpu_small.log 2>&1
echo "test exit $?"
grep -v '^\[' $OUT/test_gpu_small.log | grep -iv 'dispatch\|launch' | tail -32
make train-check > /dev/null 2>&1
$VXC ref/train_gpu_small.vx --machine machines/dev.vx -O3 2> $OUT/train_gpu_small.err | grep -v '^\[' > $OUT/vx_gpu_train.txt
echo "train exit $?"
python3 ref/compare_train.py build/llmc_train.txt $OUT/vx_gpu_train.txt --tol 0.001 | tee $OUT/compare_small.txt
