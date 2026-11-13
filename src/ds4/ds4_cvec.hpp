// src/ds4/ds4_cvec.hpp - control vectors on Maya's DeepSeek-V4.1 engine: the host side of `strata::kernels::cvec`
// (the same kernel the GLM engine uses), a port of the GLM engine's load_control_vectors
// (src/program/generate.cpp) - llama.cpp's common_control_vector_load (every file's direction.<l> times its
// scale, summed; layer 0 has none) and llama_adapter_cvec::apply with the projection-mode patch (project: the
// unit direction and its norm as the scale) - into the table cvec_upload takes.  The vector is applied after
// every steered layer's MoE write (ds4_model.cu's layer() and layer_chunk()), on top of what --ablate projects
// out of the write itself; both load in Model::load, and neither needs the other.
//
// The file format is the same GGUF as the GLM engine's and llama.cpp's: general.architecture "controlvector",
// controlvector.model_hint (this engine warns when it is not "deepseek-v41"), and direction.<l> F32 [dim] for
// the 0-based layer l (l >= 1, as in this repo's engines - the label is the layer index, so layer 0 has no
// direction; a direction past the model is ignored, as in llama.cpp).  tools/cvec_create.py makes them for
// this engine (its CVEXTRACT rows: row 0 the embedding, row l + 1 the stream after layer l's MoE write).
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "strata/artifact/gguf_reader.hpp"

namespace strata::ds4::cvec {

/// The files' summed directions and the steered range, ready for strata::kernels::cvec_upload (host side only,
/// no CUDA): `dir` is n_layer * dim (row l: the direction for the 0-based layer l), `s` is n_layer
/// (project: |d_l|, add: 1; 0 = the layer is not steered), `first_out` / `last_out` the steered range resolved
/// (llama.cpp's default 1 .. n_layer - 1, clamped to what the files cover), `summary` what the start-up log
/// reports.  `mode` 0 project, 1 add; `single` >= 0: project mode's single:L (that layer's direction for every
/// steered layer).  `first` / `last` <= 0 take the defaults.
inline bool build(const std::vector<std::pair<std::string, float>>& files, int n_layer, int dim, int mode, int first,
                  int last, int single, std::vector<float>& dir, std::vector<float>& s, int& first_out, int& last_out,
                  std::string& summary, std::string& err) {
    if (files.empty()) { err = "no control vector files"; return false; }
    if (n_layer < 2 || dim < 1) { err = "a control vector needs at least 2 layers and a hidden size"; return false; }
    std::vector<float> data((size_t) n_layer * (size_t) dim, 0.0f);
    std::vector<bool> have((size_t) n_layer, false);
    for (const auto& [path, scale] : files) {
        try {
            GgufFile f(path);
            const MetaValue* arch = f.get("general.architecture");
            if (arch == nullptr || arch->s != "controlvector") {
                err = path + ": not a control vector GGUF (general.architecture is not 'controlvector')";
                return false;
            }
            const MetaValue* hint = f.get("controlvector.model_hint");
            if (hint != nullptr && hint->s != "deepseek-v41")
                std::fprintf(stderr, "maya-ds4: %s was made for '%s', not deepseek-v41\n", path.c_str(),
                             hint->s.c_str());
            int found = 0;
            for (const TensorInfo& t : f.tensors()) {
                if (t.name.rfind("direction.", 0) != 0) continue;
                const long l = std::strtol(t.name.c_str() + 10, nullptr, 10);
                if (l < 1 || l >= n_layer) continue;   // layer 0 has no vector; past the model is ignored, as in llama.cpp
                if (t.type != 0 || t.elements() != (uint64_t) dim) {
                    err = path + ": " + t.name + " must be " + std::to_string((long long) dim) + " f32";
                    return false;
                }
                const uint64_t bytes = (uint64_t) dim * 4, fsize = f.file_size(), dstart = f.data_start();
                if (dstart > fsize || t.offset > fsize - dstart || bytes > fsize - dstart - t.offset) {   // (no sum that can wrap)
                    err = path + ": " + t.name + " lies outside the file";
                    return false;
                }
                const float* src = reinterpret_cast<const float*>(f.tensor_data(t));
                for (int64_t j = 0; j < dim; ++j) data[(size_t) l * (size_t) dim + (size_t) j] += scale * src[j];
                have[(size_t) l] = true;
                ++found;
            }
            if (found == 0) { err = path + ": no direction.<layer> tensors"; return false; }
        } catch (const std::exception& e) {
            err = path + ": " + e.what();
            return false;
        }
    }
    const int f0 = first <= 0 ? 1 : first;
    const int l0 = (last <= 0 || last >= n_layer) ? n_layer - 1 : last;
    const int sing = mode == 0 ? single : -1;   // single:L is project mode only, as in llama.cpp
    if (sing >= 0 && (sing >= n_layer || !have[(size_t) sing])) {
        err = "--cvec-dir single:" + std::to_string(sing) + ": the vector has no direction for that layer";
        return false;
    }
    dir.assign((size_t) n_layer * (size_t) dim, 0.0f);
    s.assign((size_t) n_layer, 0.0f);
    int steered = 0;
    for (int l = f0; l <= l0; ++l) {
        const int src = sing >= 0 ? sing : l;
        if (!have[(size_t) src]) continue;
        const float* d = data.data() + (size_t) src * (size_t) dim;
        if (mode == 0) {
            double nrm = 0.0;
            for (int j = 0; j < dim; ++j) nrm += (double) d[j] * (double) d[j];
            nrm = std::sqrt(nrm);
            if (nrm <= 0.0) continue;
            s[(size_t) l] = (float) nrm;
            for (int j = 0; j < dim; ++j) dir[(size_t) l * (size_t) dim + (size_t) j] = (float) (d[j] / nrm);
        } else {
            s[(size_t) l] = 1.0f;
            for (int j = 0; j < dim; ++j) dir[(size_t) l * (size_t) dim + (size_t) j] = d[j];
        }
        ++steered;
    }
    if (steered == 0) {
        err = "the control vector has no direction in layers " + std::to_string(f0) + ".." + std::to_string(l0);
        return false;
    }
    first_out = f0;
    last_out = l0;
    summary = std::string(mode == 0 ? "project" : "add") + ":" + std::to_string(f0) + "-" + std::to_string(l0) +
              (sing >= 0 ? ":single" + std::to_string(sing) : "") + (steered < (l0 - f0 + 1) ? ", " +
                                                                       std::to_string(steered) +
                                                                       " steered"
                                                                                          : "");
    return true;
}

}  // namespace strata::ds4::cvec
