// tests/glm46_kernels_test.cu - maya-glm46's own kernels against double-precision references (no model, no pack):
// the full causal attention (GLM-4.5+/4.6's GQA over BF16 K / V rows - the kernel this engine adds to maya-mm3's
// set), the q / k / v biases, the plain SwiGLU and the sigmoid router.
//
// The attention is checked over several sequence lengths, including ones that cross kTile (64 keys) and both query
// and KV head counts of the shipped models (96/8: the 12 a group the kernel is built for), with the cache rows at
// their positions and a causal mask (query t sees 0..t).  The values are random, and the CPU reference is the
// definition: softmax(q . k / sqrt(128)) . v, one head group at a time.
#include "../src/glm46/glm46_kernels.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

#include <cuda_runtime.h>

namespace {

int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-64s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

#define CK(x)                                                                                                          \
    do {                                                                                                               \
        if ((x) != cudaSuccess) {                                                                                      \
            std::fprintf(stderr, "%s failed: %s\n", #x, cudaGetErrorString(cudaGetLastError()));                       \
            return 2;                                                                                                  \
        }                                                                                                              \
    } while (0)

float h_bf16(uint16_t b) {   // BF16 -> F32, exact: the 16 bits are F32's top half
    const uint32_t bits = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}
uint16_t f2bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u = (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;   // round-to-nearest-even, NaN/Inf kept
    return (uint16_t) u;
}

// a CPU double reference for full_attn: q [T][Hq][128] F32, the BF16 rows kc / vc [pos][Hkv][128], positions 0..T-1
void ref_full_attn(const std::vector<float>& q, const std::vector<uint16_t>& kc, const std::vector<uint16_t>& vc,
                   int T, int Hq, int Hkv, std::vector<float>& out) {
    const int hd = 128, kg = Hq / Hkv;
    out.assign((size_t) T * Hq * hd, 0.0f);
    const double scale = 1.0 / std::sqrt((double) hd);
    for (int t = 0; t < T; ++t)
        for (int g = 0; g < Hkv; ++g)
            for (int h = 0; h < kg; ++h) {
                const int hq = g * kg + h;
                std::vector<double> s((size_t) t + 1), e((size_t) t + 1);
                double mx = -1e300;
                for (int kp = 0; kp <= t; ++kp) {
                    double dot = 0.0;
                    for (int d = 0; d < hd; ++d)
                        dot += (double) q[((size_t) t * Hq + hq) * hd + d] * (double) h_bf16(kc[((size_t) kp * Hkv + g) * hd + d]);
                    s[(size_t) kp] = dot * scale;
                    mx = std::max(mx, s[(size_t) kp]);
                }
                double sum = 0.0;
                for (int kp = 0; kp <= t; ++kp) {
                    e[(size_t) kp] = std::exp(s[(size_t) kp] - mx);
                    sum += e[(size_t) kp];
                }
                for (int d = 0; d < hd; ++d) {
                    double a = 0.0;
                    for (int kp = 0; kp <= t; ++kp)
                        a += e[(size_t) kp] / sum * (double) h_bf16(vc[((size_t) kp * Hkv + g) * hd + d]);
                    out[((size_t) t * Hq + hq) * hd + d] = (float) a;
                }
            }
}

int test_attn() {
    int bad = 0;
    for (int T : {1, 2, 5, 65, 130}) {
        const int Hq = 12, Hkv = 1;   // (the GLM ratio; one group keeps the reference quick and the cache small)
        const int hd = 128;
        std::vector<float> q((size_t) T * Hq * hd);
        std::vector<uint16_t> kc((size_t) T * Hkv * hd), vc((size_t) T * Hkv * hd);
        srand(1234 + T);
        for (auto& x : q) x = (float) (rand() % 2001 - 1000) / 1000.0f;
        for (auto& x : kc) x = f2bf((float) (rand() % 2001 - 1000) / 1000.0f);
        for (auto& x : vc) x = f2bf((float) (rand() % 2001 - 1000) / 1000.0f);
        float *dq = nullptr, *dout = nullptr;
        uint16_t *dk = nullptr, *dv = nullptr;
        int* dpos = nullptr;
        std::vector<int> pos((size_t) T);
        for (int i = 0; i < T; ++i) pos[(size_t) i] = i;
        CK(cudaMalloc(&dq, q.size() * 4));
        CK(cudaMalloc(&dout, q.size() * 4));
        CK(cudaMalloc(&dk, kc.size() * 2));
        CK(cudaMalloc(&dv, vc.size() * 2));
        CK(cudaMalloc(&dpos, pos.size() * 4));
        CK(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dk, kc.data(), kc.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dv, vc.data(), vc.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dpos, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice));
        glm46::k::full_attn(dq, dk, dv, dpos, T, Hq, Hkv, 0, 1.0f / std::sqrt((float) hd), dout, nullptr);
        std::vector<float> got(q.size()), want;
        CK(cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost));
        ref_full_attn(q, kc, vc, T, Hq, Hkv, want);
        double worst = 0.0;
        for (size_t i = 0; i < got.size(); ++i) worst = std::max(worst, std::fabs((double) got[i] - want[i]));
        char buf[96];
        std::snprintf(buf, sizeof buf, "full_attn T=%d, 12 query heads a KV head (worst |diff| %.2e)", T, worst);
        check(worst < 1e-4, buf);
        if (worst >= 1e-4) ++bad;
        cudaFree(dq); cudaFree(dout); cudaFree(dk); cudaFree(dv); cudaFree(dpos);
    }
    return bad;
}

// ---- a quantized K / V cache (--kv-cache): the engine's own qk_norm_rope writes it, these decode a row back (the
// reference reads exactly the values the kernel reads), and the 128-point Hadamard undoes the rotation
float e4m3_h(uint32_t b) {
    const uint32_t e = (b >> 3) & 15u, m = b & 7u;
    const float a = e == 0u ? (float) m * 0.001953125f : std::ldexp(1.0f + (float) m / 8.0f, (int) e - 7);
    return (b & 0x80u) ? -a : a;
}
float e2m1_h(uint32_t c) {
    const uint32_t m = c & 7u;
    const float a = m < 4u ? 0.5f * (float) m : m == 4u ? 2.0f : m == 5u ? 3.0f : m == 6u ? 4.0f : 6.0f;
    return (c & 8u) ? -a : a;
}
float kv_val(int fmt, const uint8_t* code, const uint8_t* sc, int d) {
    if (fmt == glm46::k::kKvI8) return (float) ((const int8_t*) code)[d] * *(const float*) sc;
    if (fmt == glm46::k::kKvFp8) return e4m3_h(code[d]) * std::ldexp(1.0f, (int) sc[d >> 5] - 127);
    const uint8_t byte = code[d >> 1];
    return e2m1_h((d & 1) ? (byte >> 4) : (byte & 15u)) * e4m3_h(sc[d >> 4]);
}
// H / sqrt(128) over 128 values (Sylvester order; its own inverse)
void had128_h(float* v) {
    float o[128];
    for (int k = 0; k < 128; ++k) {
        double a = 0.0;
        for (int j = 0; j < 128; ++j) a += __builtin_popcount((unsigned) (k & j)) & 1 ? -(double) v[j] : (double) v[j];
        o[k] = (float) (a / std::sqrt(128.0));
    }
    std::memcpy(v, o, sizeof o);
}
// attention over exact float rows, as ref_full_attn but over decoded ones
void ref_full_attn_f(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v, int T,
                     int Hq, int Hkv, std::vector<float>& out) {
    const int hd = 128, kg = Hq / Hkv;
    out.assign((size_t) T * Hq * hd, 0.0f);
    const double scale = 1.0 / std::sqrt((double) hd);
    for (int t = 0; t < T; ++t)
        for (int g = 0; g < Hkv; ++g)
            for (int h = 0; h < kg; ++h) {
                const int hq = g * kg + h;
                std::vector<double> e((size_t) t + 1);
                double mx = -1e300;
                for (int kp = 0; kp <= t; ++kp) {
                    double dot = 0.0;
                    for (int d = 0; d < hd; ++d)
                        dot += (double) q[((size_t) t * Hq + hq) * hd + d] * (double) k[((size_t) kp * Hkv + g) * hd + d];
                    e[(size_t) kp] = dot * scale;
                    mx = std::max(mx, e[(size_t) kp]);
                }
                double sum = 0.0;
                for (int kp = 0; kp <= t; ++kp) sum += (e[(size_t) kp] = std::exp(e[(size_t) kp] - mx));
                for (int d = 0; d < hd; ++d) {
                    double a = 0.0;
                    for (int kp = 0; kp <= t; ++kp) a += e[(size_t) kp] / sum * (double) v[((size_t) kp * Hkv + g) * hd + d];
                    out[((size_t) t * Hq + hq) * hd + d] = (float) a;
                }
            }
}

int test_quant_kv() {
    int bad = 0;
    const int T = 130, Hq = 12, Hkv = 1, hd = 128, n_rot = 64, half = n_rot / 2;
    for (int fmt : {glm46::k::kKvI8, glm46::k::kKvFp8, glm46::k::kKvFp4}) {
        std::vector<float> q((size_t) T * Hq * hd), k((size_t) T * Hkv * hd), v((size_t) T * Hkv * hd);
        std::vector<float> qn1((size_t) hd), kn1((size_t) hd), inv((size_t) half);
        std::vector<int> pos((size_t) T);
        srand(4321 + fmt);
        for (auto& x : q) x = (float) (rand() % 2001 - 1000) / 500.0f;
        for (auto& x : k) x = (float) (rand() % 2001 - 1000) / 500.0f;
        for (auto& x : v) x = (float) (rand() % 2001 - 1000) / 500.0f;
        for (auto& x : qn1) x = 0.5f + (float) (rand() % 101) / 100.0f;
        for (auto& x : kn1) x = 0.5f + (float) (rand() % 101) / 100.0f;
        for (int i = 0; i < half; ++i) inv[(size_t) i] = (float) std::pow(10000.0, -2.0 * i / n_rot);
        for (int i = 0; i < T; ++i) pos[(size_t) i] = i;
        const size_t rows = (size_t) T, block = glm46::k::kv_bytes(fmt, fmt, rows);
        uint8_t* hblk = (uint8_t*) std::malloc(block);
        std::memset(hblk, 0, block);
        uint16_t *kc16 = nullptr, *vc16 = nullptr;
        glm46::k::KvQ kq;
        glm46::k::kv_carve(hblk, rows, fmt, fmt, 3, kc16, vc16, kq);
        uint8_t* dblk = nullptr;
        float *dq = nullptr, *dout = nullptr, *dqn = nullptr, *dkn = nullptr, *dinv = nullptr;
        int* dpos = nullptr;
        CK(cudaMalloc(&dblk, block));
        CK(cudaMalloc(&dq, q.size() * 4));
        CK(cudaMalloc(&dout, q.size() * 4));
        CK(cudaMalloc(&dqn, (size_t) hd * 4));
        CK(cudaMalloc(&dkn, (size_t) hd * 4));
        CK(cudaMalloc(&dinv, (size_t) half * 4));
        CK(cudaMalloc(&dpos, (size_t) T * 4));
        CK(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dqn, qn1.data(), (size_t) hd * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dkn, kn1.data(), (size_t) hd * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dinv, inv.data(), (size_t) half * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dpos, pos.data(), (size_t) T * 4, cudaMemcpyHostToDevice));
        float *dk = nullptr, *dv = nullptr;
        CK(cudaMalloc(&dk, k.size() * 4));
        CK(cudaMalloc(&dv, v.size() * 4));
        CK(cudaMemcpy(dk, k.data(), k.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dv, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
        // the engine's write path: q and k in place (norm, rope), both caches at their positions, rotated, quantized
        uint16_t *kcc = nullptr, *vcc = nullptr;
        glm46::k::KvQ kqd;
        glm46::k::kv_carve(dblk, rows, fmt, fmt, 3, kcc, vcc, kqd);
        glm46::k::qk_norm_rope(dq, dk, dv, dqn, dkn, dpos, T, Hq, Hkv, hd, n_rot, dinv, 1e-5f, kcc, vcc, nullptr, 0,
                               kqd);
        std::vector<float> qt(q.size());
        CK(cudaMemcpy(qt.data(), dq, qt.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hblk, dblk, block, cudaMemcpyDeviceToHost));
        // decode the rows the kernel will read (codes / scales at kv_carve's offsets), then undo the rotation
        std::vector<float> kd((size_t) T * Hkv * hd), vd((size_t) T * Hkv * hd);
        const size_t cb = glm46::k::kv_code_bytes(fmt), sb = glm46::k::kv_scale_bytes(fmt);
        uint16_t *kch = nullptr, *vch = nullptr;
        glm46::k::KvQ kqh;   // (the host copy's own carve: kqd's pointers are the device ones)
        glm46::k::kv_carve(hblk, rows, fmt, fmt, 3, kch, vch, kqh);
        const uint8_t* kcode = kqh.kc;
        const uint8_t* ksc = kqh.ks;
        const uint8_t* vcode = kqh.vc;
        const uint8_t* vsc = kqh.vs;
        for (int r = 0; r < T; ++r) {
            float kr[128], vr[128];
            for (int d = 0; d < hd; ++d) {
                kr[d] = kv_val(fmt, kcode + ((size_t) r * Hkv) * cb, ksc + ((size_t) r * Hkv) * sb, d);
                vr[d] = kv_val(fmt, vcode + ((size_t) r * Hkv) * cb, vsc + ((size_t) r * Hkv) * sb, d);
            }
            had128_h(kr);
            had128_h(vr);
            std::memcpy(&kd[(size_t) r * hd], kr, sizeof kr);
            std::memcpy(&vd[(size_t) r * hd], vr, sizeof vr);
        }
        glm46::k::full_attn(dq, kcc, vcc, dpos, T, Hq, Hkv, 0, 1.0f / std::sqrt((float) hd), dout, nullptr,
                            glm46::k::KvHost{}, kqd);
        std::vector<float> got(q.size()), want;
        CK(cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost));
        ref_full_attn_f(qt, kd, vd, T, Hq, Hkv, want);
        double worst = 0.0;
        for (size_t i = 0; i < got.size(); ++i) worst = std::max(worst, std::fabs((double) got[i] - want[i]));
        char buf[110];
        std::snprintf(buf, sizeof buf, "full_attn, %s cache T=%d (worst |diff| %.2e)", glm46::k::kv_name(fmt), T, worst);
        check(worst < 2e-3, buf);
        if (worst >= 2e-3) ++bad;
        std::free(hblk);
        cudaFree(dblk); cudaFree(dq); cudaFree(dout); cudaFree(dqn); cudaFree(dkn); cudaFree(dinv); cudaFree(dpos);
        cudaFree(dk); cudaFree(dv);
    }
    return bad;
}

// the decode's key runs (split-K): the same attention in `splits` blocks of their own, combined - against the one
// block a KV group, and against the double reference
int test_attn_split() {
    int bad = 0;
    const int T = 130, Hq = 12, Hkv = 1, hd = 128;
    for (int splits : {2, 3, 5, 8}) {
        std::vector<float> q((size_t) T * Hq * hd);
        std::vector<uint16_t> kc((size_t) T * Hkv * hd), vc((size_t) T * Hkv * hd);
        srand(777 + splits);
        for (auto& x : q) x = (float) (rand() % 2001 - 1000) / 700.0f;
        for (auto& x : kc) x = f2bf((float) (rand() % 2001 - 1000) / 700.0f);
        for (auto& x : vc) x = f2bf((float) (rand() % 2001 - 1000) / 700.0f);
        std::vector<int> pos((size_t) T);
        for (int i = 0; i < T; ++i) pos[(size_t) i] = i;
        float *dq = nullptr, *dout = nullptr, *dpart = nullptr, *dout1 = nullptr;
        uint16_t *dk = nullptr, *dv = nullptr;
        int* dpos = nullptr;
        const size_t partn = glm46::k::attn_part_floats(T, Hkv, splits);
        CK(cudaMalloc(&dq, q.size() * 4));
        CK(cudaMalloc(&dout, q.size() * 4));
        CK(cudaMalloc(&dout1, q.size() * 4));
        CK(cudaMalloc(&dpart, partn * 4));
        CK(cudaMalloc(&dk, kc.size() * 2));
        CK(cudaMalloc(&dv, vc.size() * 2));
        CK(cudaMalloc(&dpos, pos.size() * 4));
        CK(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dk, kc.data(), kc.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dv, vc.data(), vc.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dpos, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice));
        const float sc = 1.0f / std::sqrt((float) hd);
        glm46::k::full_attn(dq, dk, dv, dpos, T, Hq, Hkv, 0, sc, dout1, nullptr);
        glm46::k::full_attn(dq, dk, dv, dpos, T, Hq, Hkv, 0, sc, dout, nullptr, glm46::k::KvHost{}, glm46::k::KvQ{},
                            splits, dpart);
        std::vector<float> one(q.size()), all(q.size());
        CK(cudaMemcpy(one.data(), dout1, one.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(all.data(), dout, all.size() * 4, cudaMemcpyDeviceToHost));
        double worst = 0.0;
        for (size_t i = 0; i < one.size(); ++i) worst = std::max(worst, std::fabs((double) one[i] - all[i]));
        char buf[110];
        std::snprintf(buf, sizeof buf, "full_attn, %d key runs T=%d (worst |diff| %.2e)", splits, T, worst);
        check(worst < 2e-6, buf);
        if (worst >= 2e-6) ++bad;
        cudaFree(dq); cudaFree(dout); cudaFree(dout1); cudaFree(dpart); cudaFree(dk); cudaFree(dv); cudaFree(dpos);
    }
    return bad;
}

int test_ops() {
    const int n = 256;
    std::vector<float> x(n), b(n), g(n), u(n);
    for (int i = 0; i < n; ++i) {
        x[(size_t) i] = (float) (i - 128) / 32.0f;
        b[(size_t) i] = 0.5f * (float) (i % 7) - 1.0f;
        g[(size_t) i] = (float) (i % 21 - 10) / 4.0f;
        u[(size_t) i] = (float) (i % 13 - 6) / 3.0f;
    }
    float *dx = nullptr, *db = nullptr, *dg = nullptr, *du = nullptr, *dh = nullptr;
    CK(cudaMalloc(&dx, n * 4));
    CK(cudaMalloc(&db, n * 4));
    CK(cudaMalloc(&dg, n * 4));
    CK(cudaMalloc(&du, n * 4));
    CK(cudaMalloc(&dh, n * 4));
    CK(cudaMemcpy(dx, x.data(), n * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, b.data(), n * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dg, g.data(), n * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(du, u.data(), n * 4, cudaMemcpyHostToDevice));
    glm46::k::bias_add(dx, db, 1, n, nullptr);
    std::vector<float> hx(n);
    CK(cudaMemcpy(hx.data(), dx, n * 4, cudaMemcpyDeviceToHost));
    double worst = 0.0;
    for (int i = 0; i < n; ++i) worst = std::max(worst, std::fabs((double) hx[(size_t) i] - (x[(size_t) i] + b[(size_t) i])));
    check(worst < 1e-6, "bias_add");
    glm46::k::swigluoai(dg, du, n, 0.0f, 0.0f, dh, nullptr, nullptr);
    std::vector<float> hs(n);
    CK(cudaMemcpy(hs.data(), dh, n * 4, cudaMemcpyDeviceToHost));
    worst = 0.0;
    for (int i = 0; i < n; ++i) {
        const double gv = g[(size_t) i], uv = u[(size_t) i];
        const double want = gv / (1.0 + std::exp(-gv)) * uv;
        worst = std::max(worst, std::fabs((double) hs[(size_t) i] - want));
    }
    check(worst < 1e-6, "swiglu (plain silu(g) * u)");
    // the router: 8 values, sigmoid + a selection bias, top-2, renormalised (the engine's scale applied later)
    std::vector<float> logits = {0.3f, -0.7f, 1.1f, 0.2f, -1.4f, 0.9f, 0.05f, -0.2f};
    std::vector<float> bias = {0.1f, 0.2f, -0.1f, 0.0f, 0.0f, 0.25f, -0.2f, 0.15f};
    float *dlg = nullptr, *dbi = nullptr;
    int* dtop = nullptr;
    float* dw = nullptr;
    CK(cudaMalloc(&dlg, 8 * 4));
    CK(cudaMalloc(&dbi, 8 * 4));
    CK(cudaMalloc(&dtop, 2 * 4));
    CK(cudaMalloc(&dw, 2 * 4));
    CK(cudaMemcpy(dlg, logits.data(), 8 * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dbi, bias.data(), 8 * 4, cudaMemcpyHostToDevice));
    glm46::k::route(dlg, dbi, 1, 8, 2, dtop, dw, nullptr);
    int top[2];
    float w[2];
    CK(cudaMemcpy(top, dtop, 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(w, dw, 8, cudaMemcpyDeviceToHost));
    std::vector<double> sg(8), sel(8);
    for (int i = 0; i < 8; ++i) sg[(size_t) i] = 1.0 / (1.0 + std::exp(-logits[(size_t) i]));
    for (int i = 0; i < 8; ++i) sel[(size_t) i] = sg[(size_t) i] + bias[(size_t) i];
    int want_top[2] = {-1, -1};
    for (int r = 0; r < 2; ++r) {
        int bi = -1;
        for (int i = 0; i < 8; ++i)
            if (i != want_top[0] && (bi < 0 || sel[(size_t) i] > sel[(size_t) bi])) bi = i;
        want_top[r] = bi;
    }
    const double sum = sg[(size_t) want_top[0]] + sg[(size_t) want_top[1]];
    char buf[96];
    const bool ok_r = top[0] == want_top[0] && top[1] == want_top[1] &&
                      std::fabs(w[0] - sg[(size_t) want_top[0]] / sum) < 1e-6 &&
                      std::fabs(w[1] - sg[(size_t) want_top[1]] / sum) < 1e-6;
    std::snprintf(buf, sizeof buf, "route (sigmoid + selection bias, top-2, renormalised): {%d,%d}", top[0], top[1]);
    check(ok_r, buf);
    cudaFree(dx); cudaFree(db); cudaFree(dg); cudaFree(du); cudaFree(dh);
    cudaFree(dlg); cudaFree(dbi); cudaFree(dtop); cudaFree(dw);
    return 0;
}

// the NextN block's input rows ([rms(emb, enorm) | rms(h, hnorm)]) and the head's argmax (the lowest id on a tie)
int test_mtp() {
    const int rows = 3, D = 128;
    std::vector<float> emb((size_t) rows * D), h((size_t) rows * D), we(D), wh(D);
    for (int r = 0; r < rows; ++r)
        for (int i = 0; i < D; ++i) {
            emb[(size_t) r * D + i] = (float) ((r * 37 + i * 11) % 53 - 26) / 8.0f;
            h[(size_t) r * D + i] = (float) ((r * 29 + i * 7) % 41 - 20) / 5.0f;
        }
    for (int i = 0; i < D; ++i) {
        we[(size_t) i] = 1.0f + 0.01f * (float) (i % 5);
        wh[(size_t) i] = 1.0f - 0.01f * (float) (i % 3);
    }
    float* de = nullptr;
    float* dh = nullptr;
    float* dwe = nullptr;
    float* dwh = nullptr;
    float* dc = nullptr;
    CK(cudaMalloc(&de, (size_t) rows * D * 4));
    CK(cudaMalloc(&dh, (size_t) rows * D * 4));
    CK(cudaMalloc(&dwe, (size_t) D * 4));
    CK(cudaMalloc(&dwh, (size_t) D * 4));
    CK(cudaMalloc(&dc, (size_t) rows * 2 * D * 4));
    CK(cudaMemcpy(de, emb.data(), (size_t) rows * D * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dh, h.data(), (size_t) rows * D * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dwe, we.data(), (size_t) D * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dwh, wh.data(), (size_t) D * 4, cudaMemcpyHostToDevice));
    glm46::k::mtp_in(de, dh, dwe, dwh, rows, D, 1e-5f, dc, nullptr);
    std::vector<float> cat((size_t) rows * 2 * D);
    CK(cudaMemcpy(cat.data(), dc, (size_t) rows * 2 * D * 4, cudaMemcpyDeviceToHost));
    double worst = 0.0;
    for (int r = 0; r < rows; ++r) {
        double se = 0, sh = 0;
        for (int i = 0; i < D; ++i) {
            se += (double) emb[(size_t) r * D + i] * emb[(size_t) r * D + i];
            sh += (double) h[(size_t) r * D + i] * h[(size_t) r * D + i];
        }
        const double re = 1.0 / std::sqrt(se / D + 1e-5), rh = 1.0 / std::sqrt(sh / D + 1e-5);
        for (int i = 0; i < D; ++i) {
            worst = std::max(worst, std::fabs((double) cat[(size_t) r * 2 * D + i] -
                                              emb[(size_t) r * D + i] * re * we[(size_t) i]));
            worst = std::max(worst, std::fabs((double) cat[(size_t) r * 2 * D + i + D] -
                                              h[(size_t) r * D + i] * rh * wh[(size_t) i]));
        }
    }
    check(worst < 1e-5, "mtp_in ([rms(emb, enorm) | rms(h, hnorm)])");
    // the argmax: a strict max, then a tie that must take the lower id
    std::vector<float> x = {-1.0f, 2.0f, 0.5f, 2.0f, -3.0f, 0.0f};
    float* dx = nullptr;
    int* di = nullptr;
    CK(cudaMalloc(&dx, x.size() * 4));
    CK(cudaMalloc(&di, 4));
    CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    glm46::k::argmax_row(dx, (int) x.size(), di, nullptr);
    int at = -1;
    CK(cudaMemcpy(&at, di, 4, cudaMemcpyDeviceToHost));
    check(at == 1, "argmax_row (a tie takes the lower id)");
    x[0] = 9.0f;
    CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    glm46::k::argmax_row(dx, (int) x.size(), di, nullptr);
    CK(cudaMemcpy(&at, di, 4, cudaMemcpyDeviceToHost));
    check(at == 0, "argmax_row (the strict max)");
    cudaFree(de); cudaFree(dh); cudaFree(dwe); cudaFree(dwh); cudaFree(dc); cudaFree(dx); cudaFree(di);
    return 0;
}

}  // namespace

int main() {
    std::printf("maya-glm46 kernels against double references\n");
    if (test_attn() != 0) ++g_fail;
    if (test_quant_kv() != 0) ++g_fail;
    if (test_attn_split() != 0) ++g_fail;
    test_ops();
    test_mtp();
    std::printf(g_fail == 0 ? "ALL PASS\n" : "%d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
