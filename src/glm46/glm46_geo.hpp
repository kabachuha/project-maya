// src/glm46/glm46_geo.hpp - maya-glm46: GLM-4.5+/GLM-4.6's (glm4moe) geometry from glm46.txt (both engine paths' -
// glm46_model.cu, glm46_tp.cu), with the checks of what the kernels cover.
//
// The GGUF (llama.cpp's glm4moe, tools/glm46/pack_gguf.py): 92/93 trunk layers (the last one the NextN block), a
// 3-layer dense lead, then 160 routed experts (8 a token, sigmoid routing with a selection bias, renormalised and
// scaled by 2.5) beside one shared expert; GQA attention (96 query and 8 KV heads, 128 a head) with a per-head
// RMSNorm on q and k and a 64-dim partial rope.
#pragma once

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "glm46_env.hpp"
#include "glm46_pack.hpp"

namespace glm46 {

struct Geo {
    int dim, n_layers, n_head, n_kv, hd, n_rot, vocab, n_exp, k_exp, ff, ff_dense, ff_shared;
    float theta, eps, scale;
    std::vector<int> moe;   // per trunk layer: 1 a MoE layer, 0 a dense one
    int bos, eos, eot;
    bool qkv_bias = true;   // glm4moe: the q / k / v projections carry a bias
    int nextn = 0;          // 1: the pack carries the NextN (MTP) block (blk.<n_layers>.*) - the native drafter
    // (inherited from maya-mm3: the MiniMax indexer / swigluoai numbers its TP and DSpark sources still name; glm4moe
    //  has neither an indexer nor a clamp, and this engine refuses --tp / loads no DSpark draft - the files stay as
    //  they were, so an arch without them compiles)
    int ix_heads = 0, ix_dim = 128, blk = 128, topk = 0, local = 1;
    float alpha = 0.0f, limit = 0.0f;
    std::vector<int> sparse;

    bool read(const Pack& p, std::string& err) {
        dim = (int) p.num("hidden_size");
        n_layers = (int) p.num("num_hidden_layers");
        n_head = (int) p.num("num_attention_heads");
        n_kv = (int) p.num("num_key_value_heads");
        hd = (int) p.num("head_dim", p.num("hidden_size") / std::max(1.0, p.num("num_attention_heads")));
        n_rot = (int) p.num("rotary_dim", hd * p.num("partial_rotary_factor", 0.5));
        vocab = (int) p.num("vocab_size");
        n_exp = (int) p.num("num_local_experts");
        k_exp = (int) p.num("num_experts_per_tok");
        ff = (int) p.num("intermediate_size");             // a routed expert's FF width
        ff_dense = (int) p.num("dense_intermediate_size"); // the dense lead's
        ff_shared = (int) p.num("shared_intermediate_size");
        theta = (float) p.num("rope_theta");
        eps = (float) p.num("rms_norm_eps");
        scale = (float) p.num("routed_scaling_factor", 1.0);
        moe = p.ints("moe_layer_freq");
        bos = (int) p.num("bos_token_id");
        eos = (int) p.num("eos_token_id");
        eot = (int) p.num("eot_token_id", eos);
        qkv_bias = p.num("qkv_bias", 1.0) != 0.0;
        nextn = (int) p.num("nextn", 0);
        const auto no = [&](const std::string& what) {
            err = "geometry not covered: " + what;
            return false;
        };
        // the kernels: 128-wide heads (the window kernel's rows), a partial rope on an even count, and kG = 12 query
        // heads a KV head (GLM-4.5 Air / 355B / 4.6: 96 / 8)
        if (hd != 128 || n_rot % 2 || n_rot > hd) return no("head_dim " + std::to_string(hd));
        if (n_kv <= 0 || n_head != 12 * n_kv) return no("query heads per KV head " + std::to_string(n_head / std::max(1, n_kv)));
        if (k_exp < 1 || k_exp > 8 || n_exp > 256) return no("experts");
        if ((int) moe.size() != n_layers) return no("moe_layer_freq (a value a trunk layer)");
        if (dim % 128 || ff % 128 || ff_dense % 128 || ff_shared % 128) return no("matrix sides");
        if (!(theta > 0) || !(eps > 0) || !(scale > 0)) return no("rope theta / rms eps / routing scale");
        return true;
    }
};

}  // namespace glm46
