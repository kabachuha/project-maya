// src/kernels/cuda/glm_lora.cu - see include/strata/kernels/glm_lora.hpp.
//
// Small deltas over big GEMMs: one dot of n_in per target per activation (the rank-r intermediate)
// and one FMA of n_out.  Rank 1 is the common case (heretic's adapters and other single-direction
// ones) and gets the unrolled kernel; ranks 2..8 unroll too, 9..32 run the runtime-rank kernel
// (the extra A traffic is invisible next to the base matmul it corrects).

#include "strata/kernels/glm_lora.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <cuda_runtime.h>

#if !defined(STRATA_USE_HIP)
#include <cuda_fp16.h>
#else
#include <hip/hip_fp16.h>
#endif

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

namespace strata::kernels::glmf {
namespace {

constexpr int kMaxLoraRankKernel = 32;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// ---- the A-steps

// single 256-thread block (8 warps) over the q8_1 blocks; one block-wide reduce, so t is written
// once (the old one-warp-per-block grid raced on t[j] += and lost most of the partial sums)
template <int R>
__global__ void __launch_bounds__(256) lora_a_q8_kernel(const float* __restrict__ A,
                                                        const block_q8_1* __restrict__ xq, int n_in,
                                                        float* __restrict__ t) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nblk = n_in / 32;
    float acc[R];
#pragma unroll
    for (int j = 0; j < R; ++j) acc[j] = 0.f;
    for (int blk = warp; blk < nblk; blk += 8) {
        const block_q8_1* b = &xq[blk];
        const float q = ((float) b->qs[lane]) * __low2float(b->ds);
#pragma unroll
        for (int j = 0; j < R; ++j) acc[j] += A[(size_t) j * n_in + blk * 32 + lane] * q;
    }
    __shared__ float sh[8][R];
#pragma unroll
    for (int j = 0; j < R; ++j) {
        const float v = warp_sum(acc[j]);   // collective: every lane must take part in the shuffle
        if (lane == 0) sh[warp][j] = v;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
#pragma unroll
        for (int j = 0; j < R; ++j) {
            float s2 = 0.f;
#pragma unroll
            for (int w = 0; w < 8; ++w) s2 += sh[w][j];
            t[j] = s2;
        }
    }
}

__global__ void __launch_bounds__(256) lora_a_q8_rt_kernel(const float* __restrict__ A, const block_q8_1* __restrict__ xq,
                                                           int n_in, int r, float* __restrict__ t) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nblk = n_in / 32;
    float acc[kMaxLoraRankKernel];
    for (int j = 0; j < r; ++j) acc[j] = 0.f;
    for (int blk = warp; blk < nblk; blk += 8) {
        const block_q8_1* b = &xq[blk];
        const float q = ((float) b->qs[lane]) * __low2float(b->ds);
        for (int j = 0; j < r; ++j) acc[j] += A[(size_t) j * n_in + blk * 32 + lane] * q;
    }
    __shared__ float sh[8][kMaxLoraRankKernel];
    for (int j = 0; j < r; ++j) {
        const float v = warp_sum(acc[j]);   // collective: every lane must take part in the shuffle
        if (lane == 0) sh[warp][j] = v;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int j = 0; j < r; ++j) {
            float s2 = 0.f;
            for (int w = 0; w < 8; ++w) s2 += sh[w][j];
            t[j] = s2;
        }
    }
}

// single 256-thread block over 1024-element chunks; one block-wide reduce, so t is written once
template <int R>
__global__ void __launch_bounds__(256) lora_a_f32_kernel(const float* __restrict__ A, const float* __restrict__ x, int n_in,
                                   float* __restrict__ t) {
    const int nchunk = n_in / 1024;
    float acc[R];
#pragma unroll
    for (int j = 0; j < R; ++j) acc[j] = 0.f;
    for (int c = 0; c < nchunk; ++c) {
        const int i0 = c * 1024 + threadIdx.x * 4;
        const float4 xv = *(const float4*) (x + i0);
#pragma unroll
        for (int j = 0; j < R; ++j) {
            const float4 a = *(const float4*) (A + (size_t) j * n_in + i0);
            acc[j] += xv.x * a.x + xv.y * a.y + xv.z * a.z + xv.w * a.w;
        }
    }
    __shared__ float sh[8][R];
#pragma unroll
    for (int j = 0; j < R; ++j) {
        const float v = warp_sum(acc[j]);   // collective: every lane must take part in the shuffle
        if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5][j] = v;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
#pragma unroll
        for (int j = 0; j < R; ++j) {
            float s2 = 0.f;
#pragma unroll
            for (int w = 0; w < 8; ++w) s2 += sh[w][j];
            t[j] = s2;
        }
    }
}

__global__ void __launch_bounds__(256) lora_a_f32_rt_kernel(const float* __restrict__ A, const float* __restrict__ x, int n_in, int r,
                                      float* __restrict__ t) {
    const int nchunk = n_in / 1024;
    float acc[kMaxLoraRankKernel];
    for (int j = 0; j < r; ++j) acc[j] = 0.f;
    for (int c = 0; c < nchunk; ++c) {
        const int i0 = c * 1024 + threadIdx.x * 4;
        const float4 xv = *(const float4*) (x + i0);
        for (int j = 0; j < r; ++j) {
            const float4 a = *(const float4*) (A + (size_t) j * n_in + i0);
            acc[j] += xv.x * a.x + xv.y * a.y + xv.z * a.z + xv.w * a.w;
        }
    }
    __shared__ float sh[8][kMaxLoraRankKernel];
    for (int j = 0; j < r; ++j) {
        const float v = warp_sum(acc[j]);   // collective: every lane must take part in the shuffle
        if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5][j] = v;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int j = 0; j < r; ++j) {
            float s2 = 0.f;
            for (int w = 0; w < 8; ++w) s2 += sh[w][j];
            t[j] = s2;
        }
    }
}

// one 256-thread block per row; a uint2 holds 4 halves (the low 16 bits of each word are the earlier
// element), so each iteration pairs 4 x-halves with 4 A floats
template <int R>
__global__ void lora_a_f16_kernel(const float* __restrict__ A, const uint16_t* __restrict__ x, int n_in,
                                  float* __restrict__ t) {
    const int row = blockIdx.x;
    const uint16_t* xr = x + (size_t) row * n_in;
    float acc[R];
#pragma unroll
    for (int j = 0; j < R; ++j) acc[j] = 0.f;
    const int n4 = n_in / 4;
    for (int i = threadIdx.x; i < n4; i += 256) {
        const uint2 h = ((const uint2*) xr)[i];
        const float v0 = f32_from_f16((uint16_t) h.x);
        const float v1 = f32_from_f16(h.x >> 16);
        const float v2 = f32_from_f16((uint16_t) h.y);
        const float v3 = f32_from_f16(h.y >> 16);
#pragma unroll
        for (int j = 0; j < R; ++j) {
            const float4 a = *(const float4*) (A + (size_t) j * n_in + 4 * (size_t) i);
            acc[j] += v0 * a.x + v1 * a.y + v2 * a.z + v3 * a.w;
        }
    }
    __shared__ float sh[8];
#pragma unroll
    for (int j = 0; j < R; ++j) {
        float v = warp_sum(acc[j]);
        if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
        __syncthreads();
        if (threadIdx.x == 0) {
            float s2 = 0.f;
#pragma unroll
            for (int w = 0; w < 8; ++w) s2 += sh[w];
            t[(size_t) row * R + j] = s2;
        }
        __syncthreads();
    }
}

__global__ void lora_a_f16_rt_kernel(const float* __restrict__ A, const uint16_t* __restrict__ x, int n_in, int r,
                                     float* __restrict__ t) {
    const int row = blockIdx.x;
    const uint16_t* xr = x + (size_t) row * n_in;
    const int n4 = n_in / 4;
    __shared__ float sh[8];
    for (int j = 0; j < r; ++j) {
        float acc = 0.f;
        for (int i = threadIdx.x; i < n4; i += 256) {
            const uint2 h = ((const uint2*) xr)[i];
            const float v0 = f32_from_f16((uint16_t) h.x);
            const float v1 = f32_from_f16(h.x >> 16);
            const float v2 = f32_from_f16((uint16_t) h.y);
            const float v3 = f32_from_f16(h.y >> 16);
            const float4 a = *(const float4*) (A + (size_t) j * n_in + 4 * (size_t) i);
            acc += v0 * a.x + v1 * a.y + v2 * a.z + v3 * a.w;
        }
        float v = warp_sum(acc);
        if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
        __syncthreads();
        if (threadIdx.x == 0) {
            float s2 = 0.f;
#pragma unroll
            for (int w = 0; w < 8; ++w) s2 += sh[w];
            t[(size_t) row * r + j] = s2;
        }
        __syncthreads();
    }
}

// ---- the B-steps

template <int R>
__global__ void lora_b_f32_kernel(const float* __restrict__ B, const float* __restrict__ t, int n_out, float scale,
                                  float* __restrict__ y) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n_out) return;
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < R; ++j) acc += B[(size_t) i * R + j] * t[j];
    y[i] += scale * acc;
}

__global__ void lora_b_f32_rt_kernel(const float* __restrict__ B, const float* __restrict__ t, int n_out, int r,
                                     float scale, float* __restrict__ y) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n_out) return;
    float acc = 0.f;
    for (int j = 0; j < r; ++j) acc += B[(size_t) i * r + j] * t[j];
    y[i] += scale * acc;
}

template <int R>
__global__ void lora_b_f32_rows_kernel(const float* __restrict__ B, const float* __restrict__ t, int n_out,
                                       float scale, float* __restrict__ y) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n_out) return;
    const float* tr = t + (size_t) blockIdx.y * R;
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < R; ++j) acc += B[(size_t) i * R + j] * tr[j];
    y[(size_t) blockIdx.y * n_out + i] += scale * acc;
}

__global__ void lora_b_f32_rows_rt_kernel(const float* __restrict__ B, const float* __restrict__ t, int n_out, int r,
                                          float scale, float* __restrict__ y) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n_out) return;
    const float* tr = t + (size_t) blockIdx.y * r;
    float acc = 0.f;
    for (int j = 0; j < r; ++j) acc += B[(size_t) i * r + j] * tr[j];
    y[(size_t) blockIdx.y * n_out + i] += scale * acc;
}

// ---- the routed experts' down

// one block per (output chunk, row): the row's plan entries are summed inside the block (8 warps split
// each entry's q8_1 row), so out gets exactly one write per (row, i) - the old one-block-per-entry grid
// raced on out[row*n_out+i] += across the entries that share a row
template <int R>
__global__ void __launch_bounds__(256) lora_moe_down_kernel(const unsigned long long* __restrict__ plan_ptr, const float* __restrict__ plan_w,
                                      const int* __restrict__ plan_id, const block_q8_1* __restrict__ hq,
                                      const float* __restrict__ A, const float* __restrict__ B, int n_exp,
                                      int n_in, int n_out, int k, float scale, int per_row, float* __restrict__ out) {
    const int row = blockIdx.y;
    const int e0 = per_row ? row * per_row : 0;
    const int e1 = per_row ? (e0 + per_row < k ? e0 + per_row : k) : k;
    const int i = blockIdx.x * 256 + threadIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nblk = n_in / 32;
    __shared__ float sw[8][R];
    __shared__ float st[R];
    float acc = 0.f;
    for (int entry = e0; entry < e1; ++entry) {
        const bool has = plan_ptr[entry] != 0ull;
        const int e = has ? plan_id[entry] : 0;
        if (lane == 0) {
#pragma unroll
            for (int j = 0; j < R; ++j) sw[warp][j] = 0.f;
        }
        __syncthreads();
        if (has) {
            float acc[R];
#pragma unroll
            for (int j = 0; j < R; ++j) acc[j] = 0.f;
            for (int b = warp; b < nblk; b += 8) {
                const block_q8_1* q = &hq[(size_t) entry * nblk + b];
                const float v = ((float) q->qs[lane]) * __low2float(q->ds);
#pragma unroll
                for (int j = 0; j < R; ++j) acc[j] += A[((size_t) e * R + j) * n_in + b * 32 + lane] * v;
            }
#pragma unroll
            for (int j = 0; j < R; ++j) {
                const float v = warp_sum(acc[j]);   // collective: every lane takes part; one writer per slot
                if (lane == 0) sw[warp][j] = v;
            }
        }
        __syncthreads();
        if (warp == 0) {
#pragma unroll
            for (int j = 0; j < R; ++j)
                if (lane == j) {
                    float v = 0.f;
#pragma unroll
                    for (int ww = 0; ww < 8; ++ww) v += sw[ww][j];
                    st[j] = v;
                }
        }
        __syncthreads();
        if (has && i < n_out) {
            const float w = plan_w[entry] * scale;
#pragma unroll
            for (int j = 0; j < R; ++j) acc += w * B[((size_t) e * n_out + i) * R + j] * st[j];
        }
        __syncthreads();
    }
    if (i < n_out) out[(size_t) row * n_out + i] += acc;
}

__global__ void __launch_bounds__(256) lora_moe_down_rt_kernel(const unsigned long long* __restrict__ plan_ptr, const float* __restrict__ plan_w,
                                         const int* __restrict__ plan_id, const block_q8_1* __restrict__ hq,
                                         const float* __restrict__ A, const float* __restrict__ B, int n_exp, int r,
                                         int n_in, int n_out, int k, float scale, int per_row, float* __restrict__ out) {
    const int row = blockIdx.y;
    const int e0 = per_row ? row * per_row : 0;
    const int e1 = per_row ? (e0 + per_row < k ? e0 + per_row : k) : k;
    const int i = blockIdx.x * 256 + threadIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nblk = n_in / 32;
    __shared__ float sw[8][kMaxLoraRankKernel];
    __shared__ float st[kMaxLoraRankKernel];
    float acc = 0.f;
    for (int entry = e0; entry < e1; ++entry) {
        const bool has = plan_ptr[entry] != 0ull;
        const int e = has ? plan_id[entry] : 0;
        if (lane == 0) {
            for (int j = 0; j < r; ++j) sw[warp][j] = 0.f;
        }
        __syncthreads();
        if (has) {
            float acc[kMaxLoraRankKernel];
            for (int j = 0; j < r; ++j) acc[j] = 0.f;
            for (int b = warp; b < nblk; b += 8) {
                const block_q8_1* q = &hq[(size_t) entry * nblk + b];
                const float v = ((float) q->qs[lane]) * __low2float(q->ds);
                for (int j = 0; j < r; ++j) acc[j] += A[((size_t) e * r + j) * n_in + b * 32 + lane] * v;
            }
            for (int j = 0; j < r; ++j) {
                const float v = warp_sum(acc[j]);   // collective: every lane takes part; one writer per slot
                if (lane == 0) sw[warp][j] = v;
            }
        }
        __syncthreads();
        if (warp == 0) {
            for (int j = 0; j < r; ++j)
                if (lane == j) {
                    float v = 0.f;
                    for (int ww = 0; ww < 8; ++ww) v += sw[ww][j];
                    st[j] = v;
                }
        }
        __syncthreads();
        if (has && i < n_out) {
            const float w = plan_w[entry] * scale;
            for (int j = 0; j < r; ++j) acc += w * B[((size_t) e * n_out + i) * r + j] * st[j];
        }
        __syncthreads();
    }
    if (i < n_out) out[(size_t) row * n_out + i] += acc;
}

// ---- the prompt path's routed experts (the swiglu h saved before the down reuses its rows)

__global__ void lora_copy_h_kernel(const float* __restrict__ src, int ld_src, int n, int n_in,
                                   float* __restrict__ dst) {
    const int64_t total = (int64_t) n * n_in;
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < total;
         i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t row = i / n_in, c = i % n_in;
        dst[i] = src[row * (int64_t) ld_src + c];
    }
}

// one 256-thread block per row: the rank-r intermediate from the row's h, then every thread its output chunk
__global__ void lora_down_rows_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                      const float* __restrict__ h, int r, int n_in, int n_out, float scale,
                                      float* __restrict__ y) {
    const int row = blockIdx.x;
    const float* hr = h + (size_t) row * n_in;
    __shared__ float sh[8];
    __shared__ float st[kMaxLoraRankKernel];
    float acc;
    for (int j = 0; j < r; ++j) {
        acc = 0.f;
        const float* ar = A + (size_t) j * n_in;
        for (int c = threadIdx.x * 8; c < n_in; c += 256 * 8) {
            const float4 x0 = *(const float4*) (hr + c);
            const float4 x1 = *(const float4*) (hr + c + 4);
            const float4 a0 = *(const float4*) (ar + c);
            const float4 a1 = *(const float4*) (ar + c + 4);
            acc += x0.x * a0.x + x0.y * a0.y + x0.z * a0.z + x0.w * a0.w + x1.x * a1.x + x1.y * a1.y + x1.z * a1.z +
                   x1.w * a1.w;
        }
        float v = warp_sum(acc);
        if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
        __syncthreads();
        if (threadIdx.x == 0) {
            float s2 = 0.f;
#pragma unroll
            for (int w = 0; w < 8; ++w) s2 += sh[w];
            st[j] = s2;
        }
        __syncthreads();
    }
    float* yr = y + (size_t) row * n_out;
    for (int i = threadIdx.x; i < n_out; i += 256) {
        float acc2 = 0.f;
        for (int j = 0; j < r; ++j) acc2 += B[(size_t) i * r + j] * st[j];
        yr[i] += scale * acc2;
    }
}

// ---- the dispatch

template <int R>
struct RankT {
    static constexpr int value = R;
};

template <class F>
void dispatch_rank(int r, F f) {
    switch (r) {
    case 1:
        f(RankT<1>{});
        break;
    case 2:
        f(RankT<2>{});
        break;
    case 3:
        f(RankT<3>{});
        break;
    case 4:
        f(RankT<4>{});
        break;
    case 5:
        f(RankT<5>{});
        break;
    case 6:
        f(RankT<6>{});
        break;
    case 7:
        f(RankT<7>{});
        break;
    case 8:
        f(RankT<8>{});
        break;
    default:
        break;   // r > 8: the callers run their runtime-rank kernel
    }
}

}  // namespace

void lora_a_f32(const float* A, const float* x, int n_in, int r, float* t, cudaStream_t s) {
    if (r <= 8)
        dispatch_rank(r, [&](auto tag) {
            constexpr int I = decltype(tag)::value;
            lora_a_f32_kernel<I><<<1, 256, 0, s>>>(A, x, n_in, t);
        });
    else
        lora_a_f32_rt_kernel<<<1, 256, 0, s>>>(A, x, n_in, r, t);
}

void lora_a_q8(const float* A, const void* xq, int n_in, int r, float* t, cudaStream_t s) {
    const block_q8_1* X = (const block_q8_1*) xq;
    if (r <= 8)
        dispatch_rank(r, [&](auto tag) {
            constexpr int I = decltype(tag)::value;
            lora_a_q8_kernel<I><<<1, 256, 0, s>>>(A, X, n_in, t);
        });
    else
        lora_a_q8_rt_kernel<<<1, 256, 0, s>>>(A, X, n_in, r, t);
}

void lora_a_f16(const float* A, const uint16_t* x, int n_in, int tn, int r, float* t, cudaStream_t s) {
    if (r <= 8)
        dispatch_rank(r, [&](auto tag) {
            constexpr int I = decltype(tag)::value;
            lora_a_f16_kernel<I><<<tn, 256, 0, s>>>(A, x, n_in, t);
        });
    else
        lora_a_f16_rt_kernel<<<tn, 256, 0, s>>>(A, x, n_in, r, t);
}

void lora_b_f32(const float* B, const float* t, int n_out, int r, float scale, float* y, cudaStream_t s) {
    if (r <= 8)
        dispatch_rank(r, [&](auto tag) {
            constexpr int I = decltype(tag)::value;
            lora_b_f32_kernel<I><<<(n_out + 255) / 256, 256, 0, s>>>(B, t, n_out, scale, y);
        });
    else
        lora_b_f32_rt_kernel<<<(n_out + 255) / 256, 256, 0, s>>>(B, t, n_out, r, scale, y);
}

void lora_b_f32_rows(const float* B, const float* t, int n_out, int tn, int r, float scale, float* y, cudaStream_t s) {
    if (r <= 8)
        dispatch_rank(r, [&](auto tag) {
            constexpr int I = decltype(tag)::value;
            lora_b_f32_rows_kernel<I><<<dim3((n_out + 255) / 256, tn), 256, 0, s>>>(B, t, n_out, scale, y);
        });
    else
        lora_b_f32_rows_rt_kernel<<<dim3((n_out + 255) / 256, tn), 256, 0, s>>>(B, t, n_out, r, scale, y);
}

void lora_moe_down(const unsigned long long* plan_ptr, const float* plan_w, const int* plan_id, const void* hq,
                   const float* A, const float* B, int n_exp, int r, int n_in, int n_out, int k, float scale,
                   int per_row, float* out, cudaStream_t s) {
    if (k <= 0) return;
    const block_q8_1* H = (const block_q8_1*) hq;
    const int nrows = per_row > 0 ? k / per_row : 1;
    if (r <= 8)
        dispatch_rank(r, [&](auto tag) {
            constexpr int I = decltype(tag)::value;
            lora_moe_down_kernel<I><<<dim3((n_out + 255) / 256, nrows), 256, 0, s>>>(plan_ptr, plan_w, plan_id, H, A, B,
                                                                                      n_exp, n_in, n_out, k, scale,
                                                                                      per_row, out);
        });
    else
        lora_moe_down_rt_kernel<<<dim3((n_out + 255) / 256, nrows), 256, 0, s>>>(plan_ptr, plan_w, plan_id, H, A, B,
                                                                                 n_exp, r, n_in, n_out, k, scale,
                                                                                 per_row, out);
}

void lora_copy_h(const float* src, int ld_src, int n, int n_in, float* dst, cudaStream_t s) {
    const int64_t total = (int64_t) n * n_in;
    const int grid = (int) std::min<int64_t>((total + 255) / 256, 4096);
    lora_copy_h_kernel<<<grid, 256, 0, s>>>(src, ld_src, n, n_in, dst);
}

void lora_down_rows_f32(const float* A, const float* B, const float* h, int n, int r, int n_in, int n_out,
                        float scale, float* y, cudaStream_t s) {
    lora_down_rows_kernel<<<n, 256, 0, s>>>(A, B, h, r, n_in, n_out, scale, y);
}

}  // namespace strata::kernels::glmf
