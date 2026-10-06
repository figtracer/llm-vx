// The C side of gpu.vx: a cuBLAS handle, device pointer offsets (Vx has no
// pointer arithmetic), zeroing, and host <-> device copies. Built with -DVX_CUDA on a
// CUDA machine. Without it, "device" memory is host memory and cublasSgemm_v2
// is a plain loop, so the GPU program runs on a Mac for testing.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef VX_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>

static void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
        exit(1);
    }
}

// VX_TF32=1 turns on TF32 tensor cores for these GEMMs, as it does for Vx's
// own (runtime/cuda_dispatch.cpp) and as llm.c's train_gpt2_fp32.cu does on
// Ampere and later. Off by default: it changes the numbers.
void* vx_cublas(void) {
    static cublasHandle_t handle = NULL;
    if (handle == NULL) {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cublasCreate failed\n");
            exit(1);
        }
        const char* tf32 = getenv("VX_TF32");
        if (tf32 && *tf32 && *tf32 != '0') {
            cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
            fprintf(stderr, "shim: cuBLAS math mode TF32\n");
        }
    }
    return handle;
}

int vx_upload_i32(int32_t* dst, const int32_t* src, int64_t n) {
    check(cudaMemcpy(dst, src, n * sizeof(int32_t), cudaMemcpyHostToDevice), "upload");
    return 0;
}

int vx_download_f32(float* dst, int64_t dst_off, const float* src, int64_t n) {
    check(cudaMemcpy(dst + dst_off, src, n * sizeof(float), cudaMemcpyDeviceToHost), "download");
    return 0;
}

int vx_synchronize(void) {
    check(cudaDeviceSynchronize(), "synchronize");
    return 0;
}

// n floats of device memory set to zero, as llm.c's CUDA version zeroes its
// gradients. A Vx kernel writes one row per thread, which is slow for this.
int vx_zero_f32(float* p, int64_t n) {
    check(cudaMemset(p, 0, n * sizeof(float)), "memset");
    return 0;
}
#else
void* vx_cublas(void) { return (void*)1; }

int vx_upload_i32(int32_t* dst, const int32_t* src, int64_t n) {
    memcpy(dst, src, n * sizeof(int32_t));
    return 0;
}

int vx_download_f32(float* dst, int64_t dst_off, const float* src, int64_t n) {
    memcpy(dst + dst_off, src, n * sizeof(float));
    return 0;
}

int vx_synchronize(void) { return 0; }

int vx_zero_f32(float* p, int64_t n) {
    memset(p, 0, n * sizeof(float));
    return 0;
}

// Column-major C = alpha * op(A) * op(B) + beta * C, with op 0 = N and 1 = T.
int cublasSgemm_v2(void* handle, int transa, int transb, int m, int n, int k,
                   const float* alpha, const float* A, int lda, const float* B, int ldb,
                   const float* beta, float* C, int ldc) {
    (void)handle;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            float acc = 0.0f;
            for (int p = 0; p < k; p++) {
                float a = transa ? A[p + (int64_t)i * lda] : A[i + (int64_t)p * lda];
                float b = transb ? B[j + (int64_t)p * ldb] : B[p + (int64_t)j * ldb];
                acc += a * b;
            }
            float* c = &C[i + (int64_t)j * ldc];
            *c = *beta == 0.0f ? *alpha * acc : *alpha * acc + *beta * *c;
        }
    }
    return 0;
}
#endif

float* vx_dev_at(float* p, int64_t off) { return p + off; }
