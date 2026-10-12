// tests/glm_lora_kernels_test.cu - the LoRA delta kernels (src/kernels/cuda/glm_lora.cu) against a scalar CPU
// reference: the A-steps (q8_1 decode, f32 slow, f16 prompt), the B-steps and the routed experts' down (the
// single-row and verify-window layouts).  Synthetic data, no model.  These pin the block-wide reduces: the
// A-steps used to race on t across blocks and the down used to race on out across a row's entries, so the
// kernels must return the exact CPU sum, not a partial one.  The shuffles inside are collective - a lane
// that skips one of them wedges the block - so this also guards the reduce structure.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/glm_lora.hpp"

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace gf = strata::kernels::glmf;
using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;

static int g_fail = 0;

static void check(cudaError_t e) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "glm_lora_kernels_test: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

static float rnd() { return (float) rand() / RAND_MAX * 2.f - 1.f; }

static void report(const char* name, const std::vector<float>& got, const std::vector<float>& ref, double rel = 2e-5) {
    float maxdiff = 0.f, maxref = 0.f;
    for (size_t i = 0; i < ref.size(); ++i) {
        maxdiff = std::fmaxf(maxdiff, fabsf(got[i] - ref[i]));
        maxref = std::fmaxf(maxref, fabsf(ref[i]));
    }
    const float tol = (float)(rel * maxref + 1e-6);
    if (maxdiff > tol) {
        ++g_fail;
        std::printf("%-26s FAIL maxdiff=%.6g tol=%.3g\n", name, (double) maxdiff, (double) tol);
    } else {
        std::printf("%-26s ok   maxdiff=%.6g\n", name, (double) maxdiff);
    }
}

template <class T> static T* dev(size_t n) {
    T* p = nullptr;
    check(cudaMalloc(&p, n * sizeof(T)));
    return p;
}

// the low half of the ds union is the scale the kernels read (__low2float); the high half is unused
static void fill_q8(block_q8_1& b) {
    const __half d = __float2half(rnd() * 0.1f);
    std::memcpy(&b.ds, &d, 2);
    for (int i = 0; i < 32; ++i) b.qs[i] = (int8_t) (rnd() * 128.f);
}
static inline float q8val(const block_q8_1& b, int i) { return ((float) b.qs[i]) * __low2float(b.ds); }

static void test_a_q8() {
    const int n_in = 8192, nblk = n_in / 32;
    for (int r : {1, 9}) {   // 9: the runtime-rank kernel
        std::vector<float> A((size_t) r * n_in);
        std::vector<block_q8_1> xq(nblk);
        for (auto& v : A) v = rnd() * 0.01f;
        for (auto& b : xq) fill_q8(b);
        std::vector<float> tref(r, 0.f);
        for (int j = 0; j < r; ++j)
            for (int b = 0; b < nblk; ++b)
                for (int i = 0; i < 32; ++i) tref[j] += A[(size_t) j * n_in + b * 32 + i] * q8val(xq[b], i);
        auto* dA = dev<float>(A.size());
        auto* dx = dev<block_q8_1>(xq.size());
        auto* dt = dev<float>(r);
        check(cudaMemcpy(dA, A.data(), A.size() * 4, cudaMemcpyHostToDevice));
        check(cudaMemcpy(dx, xq.data(), xq.size() * sizeof(block_q8_1), cudaMemcpyHostToDevice));
        gf::lora_a_q8(dA, dx, n_in, r, dt, 0);
        std::vector<float> t(r);
        check(cudaMemcpy(t.data(), dt, r * 4, cudaMemcpyDeviceToHost));
        char nm[48];
        std::snprintf(nm, sizeof nm, "lora_a_q8 n=8192 r=%d", r);
        report(nm, t, tref);
        cudaFree(dA); cudaFree(dx); cudaFree(dt);
    }
}

static void test_a_f32() {
    for (int n_in : {8192, 2048})
        for (int r : {1, 9, 32}) {   // 32: the runtime-rank kernel's register edge
            std::vector<float> A((size_t) r * n_in), x(n_in);
            for (auto& v : A) v = rnd() * 0.01f;
            for (auto& v : x) v = rnd();
            std::vector<float> tref(r, 0.f);
            for (int j = 0; j < r; ++j)
                for (int i = 0; i < n_in; ++i) tref[j] += A[(size_t) j * n_in + i] * x[i];
            auto* dA = dev<float>(A.size());
            auto* dx = dev<float>(x.size());
            auto* dt = dev<float>(r);
            check(cudaMemcpy(dA, A.data(), A.size() * 4, cudaMemcpyHostToDevice));
            check(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
            gf::lora_a_f32(dA, dx, n_in, r, dt, 0);
            std::vector<float> t(r);
            check(cudaMemcpy(t.data(), dt, r * 4, cudaMemcpyDeviceToHost));
            char nm[48];
            std::snprintf(nm, sizeof nm, "lora_a_f32 n=%d r=%d", n_in, r);
            report(nm, t, tref);
            cudaFree(dA); cudaFree(dx); cudaFree(dt);
        }
}

static void test_a_f16() {
    const int n_in = 8192;
    for (int tn : {1, 3})   // tn=1: with one row the loop must not reach past the row's end
        for (int r : {1, 3}) {
            std::vector<float> A((size_t) r * n_in);
            std::vector<uint16_t> x((size_t) tn * n_in);
            for (auto& v : A) v = rnd() * 0.01f;
            for (auto& v : x) v = f16_from_f32(rnd() * 0.5f);
            std::vector<float> tref((size_t) tn * r, 0.f);
            for (int row = 0; row < tn; ++row)
                for (int j = 0; j < r; ++j)
                    for (int i = 0; i < n_in; ++i)
                        tref[(size_t) row * r + j] += A[(size_t) j * n_in + i] * f32_from_f16(x[(size_t) row * n_in + i]);
            auto* dA = dev<float>(A.size());
            auto* dx = dev<uint16_t>(x.size());
            auto* dt = dev<float>((size_t) tn * r);
            check(cudaMemcpy(dA, A.data(), A.size() * 4, cudaMemcpyHostToDevice));
            check(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
            gf::lora_a_f16(dA, dx, n_in, tn, r, dt, 0);
            std::vector<float> t(tref.size());
            check(cudaMemcpy(t.data(), dt, t.size() * 4, cudaMemcpyDeviceToHost));
            char nm[48];
            std::snprintf(nm, sizeof nm, "lora_a_f16 n=8192 tn=%d r=%d", tn, r);
            report(nm, t, tref);
            cudaFree(dA); cudaFree(dx); cudaFree(dt);
        }
}

static void test_b() {
    const int n_out = 4096, r = 1, scale = 10;
    std::vector<float> B((size_t) n_out * r), t(r, 0.5f), y(n_out, 3.f);
    for (auto& v : B) v = rnd() * 0.1f;
    std::vector<float> yref = y;
    for (int i = 0; i < n_out; ++i) yref[i] += scale * B[i] * t[0];
    auto* dB = dev<float>(B.size());
    auto* dt = dev<float>(r);
    auto* dy = dev<float>(y.size());
    check(cudaMemcpy(dB, B.data(), B.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(dt, t.data(), r * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(dy, y.data(), y.size() * 4, cudaMemcpyHostToDevice));
    gf::lora_b_f32(dB, dt, n_out, r, scale, dy, 0);
    std::vector<float> yg(y.size());
    check(cudaMemcpy(yg.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
    report("lora_b_f32 n=4096 r=1", yg, yref);
    cudaFree(dB); cudaFree(dt); cudaFree(dy);

    // the rows step: an n_out that is not a multiple of the 256-thread chunk, rank 2
    const int n2 = 1000, r2 = 2, tn2 = 3;
    std::vector<float> B2((size_t) n2 * r2), t2((size_t) tn2 * r2), y2((size_t) tn2 * n2, -1.f);
    for (auto& v : B2) v = rnd() * 0.1f;
    for (auto& v : t2) v = rnd() * 0.1f;
    std::vector<float> y2ref = y2;
    for (int row = 0; row < tn2; ++row)
        for (int i = 0; i < n2; ++i)
            for (int j = 0; j < r2; ++j) y2ref[(size_t) row * n2 + i] += 5.f * B2[(size_t) i * r2 + j] * t2[(size_t) row * r2 + j];
    auto* dB2 = dev<float>(B2.size());
    auto* dt2 = dev<float>(t2.size());
    auto* dy2 = dev<float>(y2.size());
    check(cudaMemcpy(dB2, B2.data(), B2.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(dt2, t2.data(), t2.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(dy2, y2.data(), y2.size() * 4, cudaMemcpyHostToDevice));
    gf::lora_b_f32_rows(dB2, dt2, n2, tn2, r2, 5.f, dy2, 0);
    std::vector<float> yg2(y2.size());
    check(cudaMemcpy(yg2.data(), dy2, y2.size() * 4, cudaMemcpyDeviceToHost));
    report("lora_b_f32_rows n=1000 tn=3 r=2", yg2, y2ref);
    cudaFree(dB2); cudaFree(dt2); cudaFree(dy2);
}

static void test_moe() {
    const int n_exp = 288, n_in = 2048, nblk = n_in / 32, n_out = 4096, scale = 10;
    for (int r : {1, 9}) {
        std::vector<float> A((size_t) n_exp * r * n_in), B((size_t) n_exp * n_out * r);
        for (auto& v : A) v = rnd() * 0.01f;
        for (auto& v : B) v = rnd() * 0.1f;
        auto* dA = dev<float>(A.size());
        auto* dB = dev<float>(B.size());
        check(cudaMemcpy(dA, A.data(), A.size() * 4, cudaMemcpyHostToDevice));
        check(cudaMemcpy(dB, B.data(), B.size() * 4, cudaMemcpyHostToDevice));

        const auto run = [&](const char* nm, int k, int per_row, const std::vector<unsigned long long>& pp) {
            std::vector<float> pw(k);
            std::vector<int> pid(k);
            for (int i = 0; i < k; ++i) {
                pw[i] = rnd() * 0.5f + 0.25f;
                pid[i] = (i * 53 + k) % n_exp;
            }
            std::vector<block_q8_1> hq((size_t) k * nblk);
            for (auto& b : hq) fill_q8(b);
            const int nrows = per_row > 0 ? k / per_row : 1;
            std::vector<float> out((size_t) nrows * n_out, 0.f);
            for (int entry = 0; entry < k; ++entry) {
                if (pp[entry] == 0ull) continue;
                const int e = pid[entry];
                const int row = per_row ? entry / per_row : 0;
                std::vector<float> t(r, 0.f);
                for (int j = 0; j < r; ++j)
                    for (int b = 0; b < nblk; ++b)
                        for (int i = 0; i < 32; ++i)
                            t[j] += A[((size_t) e * r + j) * n_in + b * 32 + i] * q8val(hq[(size_t) entry * nblk + b], i);
                for (int o = 0; o < n_out; ++o)
                    for (int j = 0; j < r; ++j)
                        out[(size_t) row * n_out + o] += pw[entry] * scale * B[((size_t) e * n_out + o) * r + j] * t[j];
            }
            auto* dpp = dev<unsigned long long>(k);
            auto* dpw = dev<float>(k);
            auto* dpid = dev<int>(k);
            auto* dhq = dev<block_q8_1>((size_t) k * nblk);
            auto* dout = dev<float>(out.size());
            check(cudaMemcpy(dpp, pp.data(), k * 8, cudaMemcpyHostToDevice));
            check(cudaMemcpy(dpw, pw.data(), k * 4, cudaMemcpyHostToDevice));
            check(cudaMemcpy(dpid, pid.data(), k * 4, cudaMemcpyHostToDevice));
            check(cudaMemcpy(dhq, hq.data(), hq.size() * sizeof(block_q8_1), cudaMemcpyHostToDevice));
            check(cudaMemset(dout, 0, out.size() * 4));   // the kernel accumulates onto out
            gf::lora_moe_down(dpp, dpw, dpid, dhq, dA, dB, n_exp, r, n_in, n_out, k, scale, per_row, dout, 0);
            std::vector<float> og(out.size());
            check(cudaMemcpy(og.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost));
            report(nm, og, out);
            cudaFree(dpp); cudaFree(dpw); cudaFree(dpid); cudaFree(dhq); cudaFree(dout);
        };

        std::vector<unsigned long long> all8(8, 1ull);
        std::vector<unsigned long long> some8(8, 1ull);
        some8[2] = 0; some8[5] = 0;   // idle entries: the has = false path
        char nm[48];
        std::snprintf(nm, sizeof nm, "moe_down row k=8 r=%d", r);
        run(nm, 8, 0, all8);
        std::snprintf(nm, sizeof nm, "moe_down row k=8 r=%d 2-idle", r);
        run(nm, 8, 0, some8);
        std::snprintf(nm, sizeof nm, "moe_down win T=4 k=32 r=%d", r);
        run(nm, 32, 8, std::vector<unsigned long long>(32, 1ull));
        cudaFree(dA); cudaFree(dB);
    }
}

static void test_rows() {
    // one expert's A ([r][n_in]) and B ([n_out][r]) applied to a batch of h rows (the prefill caller's
    // contract: it launches once per expert with that expert's rows)
    const int n = 2, n_out = 4096, scale = 7;
    for (int n_in : {2048, 8192})   // 8192: the accumulation loop's multi-pass case
        for (int r : {1, 3}) {
            std::vector<float> A((size_t) r * n_in), B((size_t) n_out * r), h((size_t) n * n_in);
            for (auto& v : A) v = rnd() * 0.01f;
            for (auto& v : B) v = rnd() * 0.1f;
            for (auto& v : h) v = rnd();
            std::vector<float> y((size_t) n * n_out, 0.f);
            for (int row = 0; row < n; ++row) {
                std::vector<float> t(r, 0.f);
                for (int j = 0; j < r; ++j)
                    for (int i = 0; i < n_in; ++i)
                        t[j] += A[(size_t) j * n_in + i] * h[(size_t) row * n_in + i];
                for (int o = 0; o < n_out; ++o)
                    for (int j = 0; j < r; ++j)
                        y[(size_t) row * n_out + o] += scale * B[(size_t) o * r + j] * t[j];
            }
            auto* dA = dev<float>(A.size());
            auto* dB = dev<float>(B.size());
            auto* dh = dev<float>(h.size());
            auto* dy = dev<float>(y.size());
            check(cudaMemcpy(dA, A.data(), A.size() * 4, cudaMemcpyHostToDevice));
            check(cudaMemcpy(dB, B.data(), B.size() * 4, cudaMemcpyHostToDevice));
            check(cudaMemcpy(dh, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
            check(cudaMemset(dy, 0, y.size() * 4));
            gf::lora_down_rows_f32(dA, dB, dh, n, r, n_in, n_out, scale, dy, 0);
            std::vector<float> yg(y.size());
            check(cudaMemcpy(yg.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
            char nm[48];
            std::snprintf(nm, sizeof nm, "down_rows n=2 n_in=%d r=%d", n_in, r);
            report(nm, yg, y);
            cudaFree(dA); cudaFree(dB); cudaFree(dh); cudaFree(dy);
        }

    // copy_h: a strided source
    const int n3 = 3, nin3 = 512, ld = 1000;
    std::vector<float> src((size_t) n3 * ld);
    for (auto& v : src) v = rnd();
    std::vector<float> dst((size_t) n3 * nin3);
    for (int row = 0; row < n3; ++row)
        for (int i = 0; i < nin3; ++i) dst[(size_t) row * nin3 + i] = src[(size_t) row * ld + i];
    auto* dsrc = dev<float>(src.size());
    auto* ddst = dev<float>(dst.size());
    check(cudaMemcpy(dsrc, src.data(), src.size() * 4, cudaMemcpyHostToDevice));
    gf::lora_copy_h(dsrc, ld, n3, nin3, ddst, 0);
    std::vector<float> dg(dst.size());
    check(cudaMemcpy(dg.data(), ddst, dst.size() * 4, cudaMemcpyDeviceToHost));
    report("copy_h strided", dg, dst);
    cudaFree(dsrc); cudaFree(ddst);
}

int main() {
    srand(12345);
    test_a_q8();
    test_a_f32();
    test_a_f16();
    test_b();
    test_moe();
    test_rows();
    std::printf("glm_lora_kernels_test: %s (%d failures)\n", g_fail ? "FAIL" : "ok", g_fail);
    return g_fail ? 1 : 0;
}
