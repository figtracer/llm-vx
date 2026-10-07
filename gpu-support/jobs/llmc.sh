# Vx against llm.c's own fp32 CUDA trainer on one GPU: GPT-2 124M with random
# weights, T=1024, B=4, the same model and tokens for both. Each runs in FP32 and
# with TF32 tensor-core matmuls; Vx also with its launch-overhead fix (Vx#1331).
set -x
OUT=/workspace/out
mkdir -p build/rnd124
cc -O2 ref/make_random.c -o build/make_random && build/make_random build/rnd124 1024 50257 50304 12 12 768 40000
python3 gpu-support/entry.py build/bench_b4.vx 2 4 1024 build/rnd124
$VXC build/bench_b4.vx --machine machines/l4.vx -O3 > $OUT/vx_fp32.log 2>&1
VX_TF32=1 $VXC build/bench_b4.vx --machine machines/l4.vx -O3 > $OUT/vx_tf32.log 2>&1
grep 'step ' $OUT/vx_fp32.log $OUT/vx_tf32.log

(cd Vx && git apply ../gpu-support/vx-patches/payload-field-memchr.patch && source config.local \
  && cargo build --release --locked --bin vxc -p vxc > $OUT/cargo_memchr.log 2>&1)
echo "patched Vx build exit $?"
$VXC build/bench_b4.vx --machine machines/l4.vx -O3 > $OUT/vx_memchr_fp32.log 2>&1
VX_TF32=1 $VXC build/bench_b4.vx --machine machines/l4.vx -O3 > $OUT/vx_memchr_tf32.log 2>&1
grep 'step ' $OUT/vx_memchr_fp32.log $OUT/vx_memchr_tf32.log

# nvcc comes from NVIDIA's redistributable archives, checked against the
# manifest's sha256 (the pip wheel has only ptxas, and Debian 13's apt rejects
# NVIDIA's SHA-1 repository key). cuBLAS comes from the wheel, as for Vx.
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
grep -c 'noexcept (true);' $TK/include/crt/math_functions.h

# llm.c turns TF32 on by itself on Ampere and later; the FP32 build turns it off.
mkdir -p build/llmc_cuda && cd build/llmc_cuda
cp ../../llm.c/train_gpt2_fp32.cu train_tf32.cu
sed 's/int enable_tf32 = deviceProp.major >= 8 ? 1 : 0;/int enable_tf32 = 0;/' train_tf32.cu > train_fp32.cu
grep -c 'int enable_tf32 = 0;' train_fp32.cu
ln -sf ../rnd124/model.bin gpt2_124M.bin
ln -sf ../rnd124/tokenizer.bin gpt2_tokenizer.bin
for v in tf32 fp32; do
  $TK/bin/nvcc -allow-unsupported-compiler --threads=0 --use_fast_math -std=c++17 -O3 -arch=native -I../../llm.c \
    -I$TK/include -I/opt/cuda/include train_$v.cu -L$TK/lib -L/opt/cuda/lib64 -lcublas -lcublasLt \
    -o train_$v > $OUT/llmc_build_$v.log 2>&1
  echo "llm.c $v build exit $?"
  tail -5 $OUT/llmc_build_$v.log
  LD_LIBRARY_PATH=$TK/lib:$LD_LIBRARY_PATH ./train_$v -i ../rnd124/train_tokens.bin -j ../rnd124/val_tokens.bin \
    -v 1000 -s 1000 > $OUT/llmc_$v.log 2>&1
  echo "llm.c $v run exit $?"
  grep 'step \|TF32' $OUT/llmc_$v.log
done
cd ../..
