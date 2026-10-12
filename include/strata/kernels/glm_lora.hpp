// include/strata/kernels/glm_lora.hpp - the LoRA delta kernels (src/kernels/cuda/glm_lora.cu).
//
// One target's delta is  y += scale * B (A x),  with A row-major [r][n_in], B row-major [n_out][r],
// rank r <= 32, and x in the activation type the site feeds its base matmul.  Each A-step returns the
// rank-r intermediate t; the B-step accumulates the delta into y (the base matmul has already written
// y, on the same stream).  The routed experts' step walks a moe_route plan (glm_fast.hpp MoeDev's
// device side): entry i with plan_ptr[i] != 0 is expert plan_id[i], its activation the q8_1 row
// hq[i] of n_in values, and its delta lands on the same row its down output does (per_row: a
// batched verify window's entry (t, i) lands on row t).

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace strata::kernels::glmf {

// t[j] = sum_i A[j][i] * x[i]   (x F32, n_in values; t zeroed first, r floats)
void lora_a_f32(const float* A, const float* x, int n_in, int r, float* t, cudaStream_t s);
// the same, x a q8_1 row (n_in / 32 blocks)
void lora_a_q8(const float* A, const void* xq, int n_in, int r, float* t, cudaStream_t s);
// t[row][j] = sum_i A[j][i] * x[row][i]   (x F16 rows [tn][n_in]; t written, not accumulated)
void lora_a_f16(const float* A, const uint16_t* x, int n_in, int tn, int r, float* t, cudaStream_t s);
// y[i] += scale * sum_j B[i][j] * t[j]   (y F32, n_out values)
void lora_b_f32(const float* B, const float* t, int n_out, int r, float scale, float* y, cudaStream_t s);
// y[row][i] += scale * sum_j B[i][j] * t[row][j]
void lora_b_f32_rows(const float* B, const float* t, int n_out, int tn, int r, float scale, float* y, cudaStream_t s);
// the routed experts' down: for every plan entry i < k with plan_ptr[i] != 0, expert e = plan_id[i]:
//   out[row(i)] += plan_w[i] * scale * (B_e (A_e hq[i]))
// A [n_exp][r][n_in], B [n_exp][n_out][r], hq [k][n_in / 32] q8_1.  per_row == 0: one output row
// (the token path); per_row == 8: entry (t, i) adds to row t (the verify window).
void lora_moe_down(const unsigned long long* plan_ptr, const float* plan_w, const int* plan_id, const void* hq,
                   const float* A, const float* B, int n_exp, int r, int n_in, int n_out, int k, float scale,
                   int per_row, float* out, cudaStream_t s);
// n rows of n_in copied from src (row stride ld_src) into dst (packed n_in per row): the prompt path's swiglu h,
// saved before the down product reuses its rows
void lora_copy_h(const float* src, int ld_src, int n, int n_in, float* dst, cudaStream_t s);
// the prompt path's routed experts' down: for rr in [0, n), y[rr] += scale * (B (A h[rr])), h packed [n][n_in]
// (y row stride n_out, the down output the combine will weight with the route's)
void lora_down_rows_f32(const float* A, const float* B, const float* h, int n, int r, int n_in, int n_out, float scale,
                        float* y, cudaStream_t s);

}  // namespace strata::kernels::glmf
