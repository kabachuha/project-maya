// src/glm46/glm46_cvec.hpp - maya-glm46's control vectors (--control-vector / --control-vector-scaled): llama.cpp's
// `common_control_vector_load` (every file's `direction.<l>` times its scale, summed; layer 0 has none) and
// `llama_adapter_cvec::apply` with the projection-mode patch, into the tables `cvec_upload` takes.
//
// The GLM-4.6 vectors (gghfez/GLM-4.6-control-vectors) are llama-cvector-generator GGUFs of `direction.<l>` F32[5120]
// tensors, one file a debias or a trait direction; the recipe pairs a debias file (scale 1.0) with one trait file.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "maya/artifact/gguf_reader.hpp"
#include "maya/kernels/cvec.hpp"

namespace glm46 {

/// `files` in order (path, scale); `first0` / `last0` 0 = the defaults (1 .. L-1); `mode` 0 = project, 1 = add;
/// `single0` >= 0 the --cvec-dir single:L layer.  Uploads the tables on the current device; `summary` is the line the
/// engine prints ("add:1-91").  The layout: n_layers * n_embd directions, the vector of the 0-based layer l starting
/// at l * n_embd (layer 0 never steered), n_layers rows of scales.
inline bool cvec_load(const std::vector<std::pair<std::string, float>>& files, int first0, int last0, int mode,
                      int single0, int64_t L, int64_t N, std::string& summary, std::string& err) {
    std::vector<float> data((size_t) (L * N), 0.0f);
    std::vector<char> have((size_t) L, 0);
    for (const auto& [path, scale] : files) {
        try {
            maya::GgufFile f(path);
            const maya::MetaValue* arch = f.get("general.architecture");
            if (arch == nullptr || arch->s != "controlvector") {
                err = path + ": not a control vector GGUF (general.architecture is not 'controlvector')";
                return false;
            }
            // (the model hint llama.cpp's control vector generator wrote: its architecture name.  An older llama.cpp
            //  spells GLM-4.5 / 4.6's glm4moe "glm4_moe" - the gghfez vectors carry that - so both are the model
            //  here; a vector of another model is used too, with a note: llama.cpp applies it regardless)
            const maya::MetaValue* hint = f.get("controlvector.model_hint");
            std::string h = hint != nullptr ? hint->s : "";
            for (char& c : h) c = (char) std::tolower((unsigned char) c);
            if (!h.empty() && h != "glm4moe" && h != "glm4_moe" && h != "glm4")
                std::fprintf(stderr, "maya-glm46: NOTE: %s was made for '%s', not glm4moe - applied as llama.cpp "
                                     "would\n", path.c_str(), hint->s.c_str());
            int found = 0;
            for (const maya::TensorInfo& t : f.tensors()) {
                if (t.name.rfind("direction.", 0) != 0) continue;
                const long l = std::strtol(t.name.c_str() + 10, nullptr, 10);
                if (l < 1 || l >= L) continue;   // layer 0 has no vector; past the model is ignored, as in llama.cpp
                if (t.type != 0 || t.elements() != (uint64_t) N) {
                    err = path + ": " + t.name + " must be " + std::to_string((long long) N) + " f32";
                    return false;
                }
                const float* src = reinterpret_cast<const float*>(f.tensor_data(t));
                for (int64_t j = 0; j < N; ++j) data[(size_t) (l * N + j)] += scale * src[j];
                have[(size_t) l] = 1;
                ++found;
            }
            if (found == 0) {
                err = path + ": no direction.<layer> tensors";
                return false;
            }
        } catch (const std::exception& e) {
            err = std::string(path) + ": " + e.what();
            return false;
        }
    }
    const int first = first0 <= 0 ? 1 : first0;
    const int last = (last0 <= 0 || last0 >= L) ? (int) L - 1 : last0;
    const int single = mode == 0 ? single0 : -1;
    if (single >= 0 && (single >= L || !have[(size_t) single])) {
        err = "--cvec-dir single:" + std::to_string(single) + ": the vector has no direction for that layer";
        return false;
    }
    std::vector<float> dir((size_t) (L * N), 0.0f), s((size_t) L, 0.0f);
    int steered = 0;
    for (int64_t l = first; l <= last; ++l) {
        const int64_t src = single >= 0 ? single : l;
        if (!have[(size_t) src]) continue;
        const float* d = data.data() + (size_t) (src * N);
        if (mode == 0) {
            double nrm = 0.0;
            for (int64_t j = 0; j < N; ++j) nrm += (double) d[j] * d[j];
            nrm = std::sqrt(nrm);
            if (nrm <= 0.0) continue;
            s[(size_t) l] = (float) nrm;
            for (int64_t j = 0; j < N; ++j) dir[(size_t) (l * N + j)] = (float) (d[j] / nrm);
        } else {
            s[(size_t) l] = 1.0f;
            std::memcpy(dir.data() + (size_t) (l * N), d, (size_t) N * sizeof(float));
        }
        ++steered;
    }
    if (steered == 0) {
        err = "the control vector has no direction in layers " + std::to_string(first) + ".." + std::to_string(last);
        return false;
    }
    if (!maya::kernels::cvec_upload(dir, s, mode, first, last, N, 1, err)) return false;
    summary = std::string(mode == 0 ? "project" : "add") + ":" + std::to_string(first) + "-" + std::to_string(last) +
              (single >= 0 ? ":single" + std::to_string(single) : "");
    return true;
}

}  // namespace glm46
