// include/strata/core/glm_lora.hpp - the llama.cpp-format LoRA adapter for the glm5-next runner.
//
// A LoRA GGUF (llama.cpp's convert_lora_to_gguf.py layout, GGUF v3) holds, per target tensor, two
// factors such that the target weight W acts as W + scale * B @ A:
//   <base>.lora_a   [r][in]   (GGUF dim 0 = in, fastest)
//   <base>.lora_b   [out][r]  (GGUF dim 0 = r, fastest)
// the metadata carrying `adapter.type` ("lora") and `adapter.lora.alpha`.  The effective scale is
//   scale = user_scale * (alpha > 0 ? alpha / rank : 1)
// (llama.cpp's llama_adapter_lora::get_scale, llama-adapter.c), and the rank is the adapter's single
// rank: every lora_a shares it (llama.cpp holds one rank per adapter).
//
// The routed experts' down is a 3-D GGUF tensor, one (r x in) factor per expert:
//   blk.N.ffn_down_exps.weight.lora_a   [n_expert][r][in]   (GGUF dim 0 = in)
//   blk.N.ffn_down_exps.weight.lora_b   [n_expert][out][r]  (GGUF dim 0 = r)
// and the per-expert delta is applied with that expert's routing weight, as the expert's own down is.
//
// Targets the GLM runner applies (base tensor name -> site):
//   blk.N.attn_output.weight      the KDA / DSA mixer's out projection (into mixer)
//   blk.N.ffn_down_shexp.weight   the shared expert's down (sh_out, before the route combine)
//   blk.N.ffn_down_exps.weight    the routed experts' down (per expert, after the combine)
//   blk.N.ffn_down.weight         the dense FFN's down (layers below dense_lead)
// Anything else is refused with the tensor's name.
//
// The deltas run on the SAME activation the engine feeds the base matmul (q8_1 on the fast path, F16
// on the prompt path, F32 on the reference path), so each site moves by the direction the adapter
// learned on top of the engine's own rounding of the base GEMM - the engine's base numerics aside,
// this is the llama.cpp delta: y += scale * B (A x).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

struct Glm5Geometry;

struct Glm5Lora {
    static constexpr int kMaxRank = 32;

    struct T {
        bool has = false;
        const float *a = nullptr, *b = nullptr;   // device F32: A [r][in], B [out][r]
        int in = 0, out = 0;
        float scale = 1.0f;
    };
    struct ET {
        bool has = false;
        const float *a = nullptr, *b = nullptr;   // device F32: A [n_exp][r][in], B [n_exp][out][r]
        float *a_h = nullptr, *b_h = nullptr;     // host F32 copies: the CPU lane computes from them
        int in = 0, out = 0, n_exp = 0;
        float scale = 1.0f;
    };

    bool active = false;
    int r = 0;               // the adapter's rank (all targets share it)
    float alpha = 0.0f;      // adapter.lora.alpha (absent: 0 -> scale = user_scale)
    float user_scale = 1.0f;
    std::vector<T> attn;     // per layer, sized n_layers + nextn (the NextN block is index n_layers)
    std::vector<T> shexp;
    std::vector<T> dense;
    std::vector<ET> exps;
    float *scratch = nullptr;   // device F32 [kMaxRank]: the rank-r intermediate
    int dev = 0;
    std::string file;

    bool load(const std::string& path, const Glm5Geometry& g, float user_scale, int dev, std::string& err);
    void release();
    Glm5Lora();
    ~Glm5Lora();
    Glm5Lora(const Glm5Lora&) = delete;
    Glm5Lora& operator=(const Glm5Lora&) = delete;
};

}  // namespace strata::core
