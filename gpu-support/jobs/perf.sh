# Speed on one GPU: the small reference and training loop against llm.c, then
# GPT-2 124M (random weights, T=1024, B=4): the step time in FP32 and with
# VX_TF32=1 (mode 2), where each step's time goes (mode 3), each Vx kernel's
# device time (VX_TIME_KERNEL=1), and llm.c's own fp32 CUDA trainer on the same
# model and data.
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
python3 gpu-support/entry.py build/phases_b4.vx 3 4 1024 build/rnd124
$VXC build/bench_b4.vx --machine machines/l4.vx -O3 > $OUT/bench_b4.log 2>&1
echo "B=4 exit $?"
grep 'step ' $OUT/bench_b4.log
VX_TF32=1 $VXC build/bench_b4.vx --machine machines/l4.vx -O3 > $OUT/bench_b4_tf32.log 2>&1
echo "B=4 TF32 exit $?"
grep 'step \|TF32' $OUT/bench_b4_tf32.log
$VXC build/phases_b4.vx --machine machines/l4.vx -O3 > $OUT/phases_b4.log 2>&1
grep -A1 'step 4' $OUT/phases_b4.log
VX_TIME_KERNEL=1 $VXC build/bench_b4.vx --machine machines/l4.vx -O3 > build/bench_b4_timed.log 2>&1
$VXC build/bench_b4.vx --machine machines/l4.vx -O3 --action emit-llvm > build/bench_b4.ll 2>/dev/null
python3 gpu-support/kernel_times.py build/bench_b4.ll build/bench_b4_timed.log --steps 5 | tee $OUT/kernel_times.txt

# llm.c's fp32 CUDA trainer, built with nvcc from NVIDIA's apt repository (the
# pip wheel has only ptxas). It turns TF32 on by itself on Ampere and later, so
# compare it with the VX_TF32=1 run. Nothing here may stop the job.
set +e
wget -qO /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb \
  && dpkg -i /tmp/cuda-keyring.deb > /dev/null && apt-get update -qq
NVCC_PKG=$(apt-cache search --names-only '^cuda-nvcc-12-[0-9]+$' | awk '{print $1}' | sort -V | tail -1)
DEV_PKG=${NVCC_PKG/nvcc/cudart-dev}
echo "installing $NVCC_PKG $DEV_PKG"
apt-get install -y -qq --no-install-recommends $NVCC_PKG $DEV_PKG > $OUT/llmc_cuda_build.log 2>&1
NVCC=$(ls -d /usr/local/cuda-12*/bin/nvcc 2>/dev/null | sort -V | tail -1)
if [ -n "$NVCC" ]; then
  TK=$(dirname $(dirname $NVCC))
  mkdir -p build/llmc_cuda && cd build/llmc_cuda
  ln -sf ../rnd124/model.bin gpt2_124M.bin
  ln -sf ../rnd124/tokenizer.bin gpt2_tokenizer.bin
  $NVCC -allow-unsupported-compiler --threads=0 --use_fast_math -std=c++17 -O3 -arch=native -I../../llm.c -I$TK/include -I/opt/cuda/include \
    ../../llm.c/train_gpt2_fp32.cu -L$TK/lib64 -L/opt/cuda/lib64 -lcublas -lcublasLt \
    -o train_gpt2fp32cu >> $OUT/llmc_cuda_build.log 2>&1
  echo "llm.c build exit $?"
  tail -5 $OUT/llmc_cuda_build.log
  LD_LIBRARY_PATH=$TK/lib64:$LD_LIBRARY_PATH ./train_gpt2fp32cu -i ../rnd124/train_tokens.bin -j ../rnd124/val_tokens.bin \
    -v 1000 -s 1000 > $OUT/llmc_cuda.log 2>&1
  echo "llm.c run exit $?"
  grep 'step \|TF32' $OUT/llmc_cuda.log | tail -12
  cd ../..
else
  echo "no nvcc installed"
  tail -20 $OUT/llmc_cuda_build.log
fi
