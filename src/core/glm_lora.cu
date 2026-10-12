// src/core/glm_lora.cu - loads a llama.cpp-format LoRA GGUF into device (and host, for the CPU
// lane) F32 tables, validated against the model's geometry.  See include/strata/core/glm_lora.hpp
// for the format contract; the scale is llama.cpp's get_scale.
//
// The file's bytes are already in the layout the kernels want (GGUF dim 0 fastest == the row-major
// [r][in] / [out][r] / [n_exp][r][in] / [n_exp][out][r] orders), so loading is a per-element
// F16/BF16 -> F32 widening copy, no transposes.

#include "strata/core/glm_lora.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

namespace strata::core {

Glm5Lora::Glm5Lora() = default;

Glm5Lora::~Glm5Lora() {
    release();
}

void Glm5Lora::release() {
    if (!active && scratch == nullptr) return;
    if (scratch) {
        cudaSetDevice(dev);
        cudaFree(scratch);
        scratch = nullptr;
    }
    for (auto& t : attn) {
        if (t.a) cudaFree(const_cast<float*>(t.a));
        if (t.b) cudaFree(const_cast<float*>(t.b));
        t = T{};
    }
    for (auto& t : shexp) {
        if (t.a) cudaFree(const_cast<float*>(t.a));
        if (t.b) cudaFree(const_cast<float*>(t.b));
        t = T{};
    }
    for (auto& t : dense) {
        if (t.a) cudaFree(const_cast<float*>(t.a));
        if (t.b) cudaFree(const_cast<float*>(t.b));
        t = T{};
    }
    for (auto& t : exps) {
        if (t.a) cudaFree(const_cast<float*>(t.a));
        if (t.b) cudaFree(const_cast<float*>(t.b));
        if (t.a_h) std::free(t.a_h);
        if (t.b_h) std::free(t.b_h);
        t = ET{};
    }
    attn.clear();
    shexp.clear();
    dense.clear();
    exps.clear();
    active = false;
    r = 0;
}

bool Glm5Lora::load(const std::string& path, const Glm5Geometry& g, float us, int dev, std::string& err) {
    release();
    file = path;
    user_scale = us;
    GgufFile gf(path);   // throws a std::runtime_error with the file's fault on a bad header
    const MetaValue* arch = gf.get("general.architecture");
    if (arch == nullptr || arch->type != MetaType::STRING || arch->s != "glm5next") {
        err = "lora: " + path + ": general.architecture must be \"glm5next\" (this is a glm5-next adapter)";
        return false;
    }
    const MetaValue* at = gf.get("adapter.type");
    if (at == nullptr || at->type != MetaType::STRING || at->s != "lora") {
        err = "lora: " + path + ": adapter.type must be \"lora\" (llama.cpp's LoRA GGUF)";
        return false;
    }
    alpha = 0.0f;
    if (const MetaValue* al = gf.get("adapter.lora.alpha")) alpha = (float) al->num();

    const int n = g.n_layers + g.nextn;
    attn.assign(n, T{});
    shexp.assign(n, T{});
    dense.assign(n, T{});
    exps.assign(n, ET{});

    struct Pair {
        const TensorInfo* a;
        const TensorInfo* b;
        int il;
        int which;   // 0 attn, 1 shexp, 2 dense, 3 exps
    };
    std::vector<Pair> pairs;
    r = 0;
    for (const TensorInfo& t : gf.tensors()) {
        if (t.name.size() <= 7 || t.name.compare(t.name.size() - 7, 7, ".lora_a") != 0) continue;
        const std::string base = t.name.substr(0, t.name.size() - 7);
        if (base.rfind("blk.", 0) != 0) {
            err = "lora: " + t.name + " is not a blk.N target";
            return false;
        }
        const size_t dot = base.find('.', 4);
        if (dot == std::string::npos) {
            err = "lora: malformed tensor name " + t.name;
            return false;
        }
        const int il = std::atoi(base.c_str() + 4);
        if (il < 0 || il >= n) {
            err = "lora: " + t.name + ": layer " + std::to_string(il) + " is out of range (0.." +
                  std::to_string(n - 1) + ")";
            return false;
        }
        const std::string rest = base.substr(dot + 1);
        int which = -1;
        if (rest == "attn_output.weight")
            which = 0;
        else if (rest == "ffn_down_shexp.weight")
            which = 1;
        else if (rest == "ffn_down.weight")
            which = 2;
        else if (rest == "ffn_down_exps.weight")
            which = 3;
        if (which < 0) {
            err = "lora: " + t.name + ": unsupported target " + rest +
                  " (this runner applies attn_output, ffn_down, ffn_down_shexp and ffn_down_exps only)";
            return false;
        }
        const TensorInfo* b = gf.find(base + ".lora_b");
        if (b == nullptr) {
            err = "lora: " + base + " has no .lora_b";
            return false;
        }
        for (const TensorInfo* p : {&t, b}) {
            if (p->type != 0 && p->type != 1 && p->type != 30) {   // F32, F16, BF16 (the reader's ids)
                err = "lora: " + p->name + " must be F32, F16 or BF16 (got " + p->type_name() + ")";
                return false;
            }
        }
        // GGUF dim 0 fastest: lora_a [in, r, (n_exp)], lora_b [r, out, (n_exp)]
        int rank = 0, in = 0, out = 0, n_exp = 1;
        if (t.shape.size() == 2) {
            in = (int) t.shape[0];
            rank = (int) t.shape[1];
            out = (int) b->shape[1];
        } else if (t.shape.size() == 3) {
            in = (int) t.shape[0];
            rank = (int) t.shape[1];
            n_exp = (int) t.shape[2];
            out = (int) b->shape[1];
        } else {
            err = "lora: " + t.name + " must be 2-D or 3-D (got " + std::to_string(t.shape.size()) + "-D)";
            return false;
        }
        if (b->shape.size() != t.shape.size() || b->shape[0] != (uint64_t) rank ||
            (t.shape.size() == 3 && b->shape[2] != (uint64_t) n_exp)) {
            err = "lora: " + base + ".lora_b does not match " + base + ".lora_a";
            return false;
        }
        if (r == 0)
            r = rank;
        else if (r != rank) {
            err = "lora: " + t.name + " has rank " + std::to_string(rank) + " but the adapter's rank is " +
                  std::to_string(r) + " (one rank per adapter, as llama.cpp requires)";
            return false;
        }
        // the geometry: the delta must fit the site it lands on
        if (which == 0) {
            const int want = g.is_recr(il) ? g.d_inner() : g.n_head * g.v_head;
            if (in != want || out != g.n_embd) {
                err = "lora: " + t.name + " is " + std::to_string(in) + "x" + std::to_string(out) +
                      " but layer " + std::to_string(il) + "'s attn_output is " + std::to_string(want) + "x" +
                      std::to_string(g.n_embd);
                return false;
            }
        } else if (which == 1) {
            if (il < g.dense_lead || in != g.n_ff_exp * g.n_shared || out != g.n_embd) {
                err = "lora: " + t.name + " does not fit layer " + std::to_string(il) + "'s shared expert";
                return false;
            }
        } else if (which == 2) {
            if (il >= g.dense_lead || in != g.n_ff_dense || out != g.n_embd) {
                err = "lora: " + t.name + " does not fit layer " + std::to_string(il) + "'s dense FFN";
                return false;
            }
        } else {
            if (il < g.dense_lead || in != g.n_ff_exp || out != g.n_embd || n_exp != g.n_expert) {
                err = "lora: " + t.name + " is " + std::to_string(n_exp) + " experts of " + std::to_string(in) +
                      "x" + std::to_string(out) + " but layer " + std::to_string(il) + " routes " +
                      std::to_string(g.n_expert) + " experts of " + std::to_string(g.n_ff_exp) + "x" +
                      std::to_string(g.n_embd);
                return false;
            }
        }
        pairs.push_back({&t, b, il, which});
    }
    if (pairs.empty()) {
        err = "lora: " + path + " holds no .lora_a tensors";
        return false;
    }
    if (r > kMaxRank) {
        err = "lora: rank " + std::to_string(r) + " exceeds the supported " + std::to_string(kMaxRank);
        return false;
    }

    cudaSetDevice(dev);
    this->dev = dev;
    const float scale = us * (alpha > 0.0f ? alpha / (float) r : 1.0f);
    for (const Pair& pr : pairs) {
        const size_t na = pr.a->elements(), nb = pr.b->elements();
        float *da = nullptr, *db = nullptr, *ha = nullptr, *hb = nullptr;
        // F32: memcpy the whole tensor; F16: widen per element; BF16: the top half of its f32 image.
        // llama.cpp LoRA adapters are F16 (or F32); anything else is rejected, not guessed
        auto widen_copy = [&](const TensorInfo& t, float* h) {
            const uint8_t* p = gf.tensor_data(t);
            if (t.type == 0) {
                std::memcpy(h, p, t.elements() * sizeof(float));
            } else if (t.type == 1) {
                const uint16_t* p16 = (const uint16_t*) p;
                for (size_t i = 0; i < t.elements(); ++i) h[i] = strata::kernels::f32_from_f16(p16[i]);
            } else if (t.type == 30) {   // BF16 (the reader's ggml type id)
                const uint16_t* p16 = (const uint16_t*) p;
                for (size_t i = 0; i < t.elements(); ++i) {
                    const uint32_t u = (uint32_t) p16[i] << 16;
                    std::memcpy(&h[i], &u, 4);
                }
            } else {
                err = "lora: " + t.name + " has type " + t.type_name() + " (only F32/F16/BF16 adapters are supported)";
                return false;
            }
            return true;
        };
        std::vector<float> ha_v(na), hb_v(nb);
        if (!widen_copy(*pr.a, ha_v.data())) return false;
        if (!widen_copy(*pr.b, hb_v.data())) return false;
        for (const std::vector<float>& v : {ha_v, hb_v})
            for (float x : v)
                if (!std::isfinite(x)) {
                    err = "lora: " + pr.a->name + " holds a non-finite value (a corrupt or truncated adapter?)";
                    return false;
                }
        if (pr.which == 3) {   // the CPU lane keeps host copies
            ha = (float*) std::malloc(na * sizeof(float));
            hb = (float*) std::malloc(pr.b->elements() * sizeof(float));
            std::memcpy(ha, ha_v.data(), na * sizeof(float));
            std::memcpy(hb, hb_v.data(), pr.b->elements() * sizeof(float));
        }
        cudaError_t e = cudaMalloc(&da, na * sizeof(float));
        e = e == cudaSuccess ? cudaMemcpyAsync(da, ha_v.data(), na * sizeof(float), cudaMemcpyHostToDevice, 0) : e;
        if (e == cudaSuccess) e = cudaMalloc(&db, pr.b->elements() * sizeof(float));
        if (e == cudaSuccess)
            e = cudaMemcpyAsync(db, hb_v.data(), pr.b->elements() * sizeof(float), cudaMemcpyHostToDevice, 0);
        if (e != cudaSuccess) {
            if (da) cudaFree(da);
            if (db) cudaFree(db);
            if (ha) std::free(ha);
            if (hb) std::free(hb);
            err = "lora: uploading " + pr.a->name + ": " + cudaGetErrorString(e);
            return false;
        }
        const float* A = da;
        const float* B = db;
        switch (pr.which) {
        case 0:
            attn[pr.il] = T{true, A, B, (int) pr.a->shape[0], (int) pr.b->shape[1], scale};
            break;
        case 1:
            shexp[pr.il] = T{true, A, B, (int) pr.a->shape[0], (int) pr.b->shape[1], scale};
            break;
        case 2:
            dense[pr.il] = T{true, A, B, (int) pr.a->shape[0], (int) pr.b->shape[1], scale};
            break;
        default:
            exps[pr.il] = ET{true, A, B, ha, hb, (int) pr.a->shape[0], (int) pr.b->shape[1], (int) pr.a->shape[2],
                             scale};
            break;
        }
    }
    if (cudaMalloc(&scratch, kMaxRank * sizeof(float)) != cudaSuccess) {
        err = "lora: scratch cudaMalloc failed";
        return false;
    }
    active = true;
    return true;
}

}  // namespace strata::core
