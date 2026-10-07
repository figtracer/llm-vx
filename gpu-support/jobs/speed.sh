# Speed on one GPU, after attention moved to cuBLAS: the small reference and
# training loop against llm.c on the GPU, then GPT-2 124M (random weights,
# T=1024, B=4). First the step time for a few widths of the narrow tensors (AW
# for the attention matrices, EW for the elementwise ones); then, with the
# fastest, FP32 and TF32 step times, where each step's time goes (mode 3) and
# each Vx kernel's device time; the same with Vx's launch-overhead fix
# (Vx#1331); and llm.c's own fp32 CUDA trainer on the same model and data.
set -x
OUT=/workspace/out
$VXC ref/test_gpu_small.vx --machine machines/dev.vx -O3 2>/dev/null | grep -v '^\[' > $OUT/test_gpu_small.log
echo "test exit $?"
tail -3 $OUT/test_gpu_small.log
$VXC ref/train_gpu_small.vx --machine machines/dev.vx -O3 2>/dev/null | grep -v '^\[' > $OUT/vx_gpu_train.txt
make train-check >/dev/null 2>&1; python3 ref/compare_train.py build/llmc_train.txt $OUT/vx_gpu_train.txt --tol 0.001 | tee $OUT/compare_small.txt
mkdir -p build/rnd124
cc -O2 ref/make_random.c -o build/make_random && build/make_random build/rnd124 1024 50257 50304 12 12 768 40000

# Stop before renting time on a kernel that would run on one thread.
python3 gpu-support/entry.py build/w_check.vx 2 4 1024 build/rnd124
$VXC build/w_check.vx --machine machines/l4.vx -O3 --action emit-llvm > build/w_check.ll 2>/dev/null
python3 gpu-support/serial_kernels.py build/w_check.ll > $OUT/serial_kernels.txt
serial=$?
cat $OUT/serial_kernels.txt
[ $serial = 0 ] || exit 1

# Mean of steps 1 to 4 in ms.
mean_step() { grep -o 'step [1-4]: .*took [0-9.]*' "$1" | grep -o '[0-9.]*$' | awk '{s += $1} END {if (NR) printf "%.1f", s / NR; else print "fail"}'; }
for w in "4 2" "8 2" "16 2" "8 4"; do
  set -- $w
  python3 gpu-support/entry.py build/w$1_$2.vx 2 4 1024 build/rnd124 --aw $1 --ew $2
  timeout 300 $VXC build/w$1_$2.vx --machine machines/l4.vx -O3 > $OUT/widths_aw$1_ew$2.log 2>&1
  echo "AW=$1 EW=$2: $(mean_step $OUT/widths_aw$1_ew$2.log) ms" | tee -a $OUT/widths.txt
done
read -r AW EW < <(sed -n 's/AW=\([0-9]*\) EW=\([0-9]*\): \([0-9.]*\) ms/\3 \1 \2/p' $OUT/widths.txt | sort -n | head -1 | cut -d' ' -f2-)
echo "fastest: AW=$AW EW=$EW" | tee -a $OUT/widths.txt

python3 gpu-support/entry.py build/bench_b4.vx 2 4 1024 build/rnd124 --aw $AW --ew $EW
python3 gpu-support/entry.py build/phases_b4.vx 3 4 1024 build/rnd124 --aw $AW --ew $EW
run_all() {
  $VXC build/bench_b4.vx --machine machines/l4.vx -O3 > $OUT/vx_$1fp32.log 2>&1
  VX_TF32=1 $VXC build/bench_b4.vx --machine machines/l4.vx -O3 > $OUT/vx_$1tf32.log 2>&1
  grep 'step ' $OUT/vx_$1fp32.log $OUT/vx_$1tf32.log
  $VXC build/phases_b4.vx --machine machines/l4.vx -O3 > $OUT/phases_$1b4.log 2>&1
  grep -A1 'step 4' $OUT/phases_$1b4.log
  VX_TIME_KERNEL=1 $VXC build/bench_b4.vx --machine machines/l4.vx -O3 > build/bench_b4_timed.log 2>&1
  $VXC build/bench_b4.vx --machine machines/l4.vx -O3 --action emit-llvm > build/bench_b4.ll 2>/dev/null
  python3 gpu-support/kernel_times.py build/bench_b4.ll build/bench_b4_timed.log --steps 5 | tee $OUT/kernel_times_$1b4.txt
}
run_all ""

# The launch-overhead fix proposed to Vx: rebuild Vx with
# gpu-support/vx-patches/payload-field-memchr.patch and time it all again.
set +e
(cd Vx && git apply ../gpu-support/vx-patches/payload-field-memchr.patch && source config.local \
  && cargo build --release --locked --bin vxc -p vxc > $OUT/cargo_memchr.log 2>&1)
echo "patched Vx build exit $?"
tail -1 $OUT/cargo_memchr.log
run_all memchr_

# llm.c's fp32 CUDA trainer on the same model and data. nvcc comes from NVIDIA's
# redistributable archives, checked against the manifest's sha256 (the pip wheel
# has only ptxas, and Debian 13's apt rejects NVIDIA's SHA-1 repository key).
# cuBLAS comes from the wheel, as for Vx.
apt-get install -y -qq xz-utils > /dev/null 2>&1
REDIST=https://developer.download.nvidia.com/compute/cuda/redist
MAN=$(curl -s $REDIST/ | grep -o 'redistrib_12\.[0-9.]*\.json' | sort -uV | tail -1)
curl -s -o /tmp/redist.json $REDIST/$MAN
TK=/opt/cudatk
mkdir -p $TK
for comp in cuda_nvcc cuda_cudart cuda_cccl; do
  read -r rel sha < <(python3 -c "import json; e=json.load(open('/tmp/redist.json'))['$comp']['linux-x86_64']; print(e['relative_path'], e['sha256'])")
  curl -s -o /tmp/$comp.tar.xz $REDIST/$rel
  echo "$sha  /tmp/$comp.tar.xz" | sha256sum -c --quiet && tar xJf /tmp/$comp.tar.xz -C $TK --strip-components=1
done
# glibc 2.41 declares sinpi, sinpif, cospi and cospif noexcept, and CUDA 12's
# headers declare them without it, which nvcc rejects as a conflict.
sed -i -E 's/^(extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ (double|float) +(sinpi|sinpif|cospi|cospif)\((double|float) x\));/\1 noexcept (true);/' \
  $TK/include/crt/math_functions.h

# llm.c turns TF32 on by itself on Ampere and later; the FP32 build turns it off.
mkdir -p build/llmc_cuda && cd build/llmc_cuda
cp ../../llm.c/train_gpt2_fp32.cu train_tf32.cu
sed 's/int enable_tf32 = deviceProp.major >= 8 ? 1 : 0;/int enable_tf32 = 0;/' train_tf32.cu > train_fp32.cu
ln -sf ../rnd124/model.bin gpt2_124M.bin
ln -sf ../rnd124/tokenizer.bin gpt2_tokenizer.bin
for v in tf32 fp32; do
  $TK/bin/nvcc -allow-unsupported-compiler --threads=0 --use_fast_math -std=c++17 -O3 -arch=native -I../../llm.c \
    -I$TK/include -I/opt/cuda/include train_$v.cu -L$TK/lib -L/opt/cuda/lib64 -lcublas -lcublasLt \
    -o train_$v > $OUT/llmc_build_$v.log 2>&1
  echo "llm.c $v build exit $?"
  LD_LIBRARY_PATH=$TK/lib:$LD_LIBRARY_PATH ./train_$v -i ../rnd124/train_tokens.bin -j ../rnd124/val_tokens.bin \
    -v 1000 -s 1000 > $OUT/llmc_$v.log 2>&1
  echo "llm.c $v run exit $?"
  grep 'step \|TF32' $OUT/llmc_$v.log
done
cd ../..
