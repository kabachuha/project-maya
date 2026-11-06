// src/core/glm_cvec_extract_test.cpp - the control-vector extraction (Glm5Model::cvec_extract, the
// CVEXTRACT serve command behind tools/cvec_create.py) against the 45-layer big fixture.  It runs the
// reference path over a prompt and writes the LAST token's mean-over-streams residual - row 0 the
// embedding, row l + 1 after layer l - as (n_layers + 1) * n_embd F32, so the python side can form the
// per-layer deltas.  The gates:
//
//   1. SHAPE: the file is exactly (n_layers + 1) * n_embd floats, all finite.
//   2. PER-PROMPT RESET: two extractions of the same prompt are bitwise identical (the serve loop reuses
//      one model across CVEXTRACT lines, so each extraction must start a fresh sequence).
//   3. NON-TRIVIAL: the captured residual is not all ~zero (something was recorded).
//   4. LAST-TOKEN SENSITIVE: swapping only the final token moves the last layer's delta (row L - row L - 1)
//      - the capture is at the right position, not a stale row.
//
// No oracle: the f64 dump under data/glm5-big-dumps is logits only, not the per-layer residual.

#include "strata/core/glm_model.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

void require(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "glm_cvec_extract_test: %s\n", what.c_str());
        std::exit(1);
    }
}

std::vector<float> read_f32(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    require(f != nullptr, "cannot open " + path);
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<float> v(sz / 4);
    if (!v.empty()) require(std::fread(v.data(), sizeof(float), v.size(), f) == v.size(), "short read on " + path);
    std::fclose(f);
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string gguf = argc > 1 ? argv[1] : "glm5-big-fixture.gguf";
    const std::string o1 = argc > 2 ? argv[2] : "glm_cvec_ex_a.bin";
    const std::string o2 = argc > 3 ? argv[3] : "glm_cvec_ex_b.bin";
    const std::string o3 = argc > 4 ? argv[4] : "glm_cvec_ex_c.bin";

    strata::core::Glm5Model model;
    std::string err;
    require(model.load(gguf, 1024, err), "load: " + err);
    require(!model.fast(), "the plain GGUF load runs the reference path");
    const int L = model.geometry().n_layers, N = model.geometry().n_embd;
    require(L == 45, "the big fixture has 45 layers");
    const size_t R = (size_t) (L + 1) * (size_t) N;   // (n_layers + 1) rows of n_embd

    std::vector<int32_t> toks;
    for (int i = 0; i < 32; ++i) toks.push_back((int32_t) ((i * 7) % 511 + 1));

    require(model.cvec_extract(toks, o1, err), "extract a: " + err);
    require(model.cvec_extract(toks, o2, err), "extract b: " + err);
    const std::vector<float> a = read_f32(o1), b = read_f32(o2);
    require(a.size() == R, "the file is (n_layers + 1) * n_embd floats");
    require(a == b, "two extractions of the same prompt are bitwise identical (per-prompt reset)");
    for (float x : a) require(std::isfinite(x), "a captured residual is not finite");
    bool any = false;
    for (float x : a)
        if (std::fabs(x) > 1e-3f) { any = true; break; }
    require(any, "the captured residual is all ~zero (nothing was recorded)");

    // last-token sensitivity: swap only the final token; the last layer's delta must move
    const int32_t last0 = toks.back();
    toks.back() = (int32_t) ((last0 + 13) % 511 + 1);
    require(model.cvec_extract(toks, o3, err), "extract c: " + err);
    const std::vector<float> c = read_f32(o3);
    require(c.size() == R, "the file (c) is (n_layers + 1) * n_embd floats");
    double shift = 0.0;
    for (int j = 0; j < N; ++j) {
        const float da = a[(size_t) L * N + j] - a[(size_t) (L - 1) * N + j];
        const float dc = c[(size_t) L * N + j] - c[(size_t) (L - 1) * N + j];
        shift = std::max(shift, (double) std::fabs(da - dc));
    }
    require(shift > 1e-3, "swapping the last token did not move the last layer's delta (wrong capture position)");
    std::printf("glm_cvec_extract_test: PASS (%zu floats, n_embd %d, deterministic, last-token shift %.3e)\n", R, N,
                shift);
    return 0;
}
