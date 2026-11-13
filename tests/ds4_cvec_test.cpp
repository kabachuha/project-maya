// tests/ds4_cvec_test.cpp - ds4_cvec.hpp's build() on synthetic GGUFs (no model, no GPU): the host side of
// --control-vector for the DeepSeek engine.
//
// What the loader sums and where it places it (the GLM engine's load_control_vectors, the same conventions):
//   * direction.<l> is the 0-based layer l (l >= 1; layer 0 has no direction, past the model is ignored);
//   * several files' directions times their scales, summed per layer;
//   * project: the unit direction, the norm as the scale; add: the direction as is, scale 1;
//   * the steered range (default 1 .. n_layer - 1, clamped by --control-vector-layer-range), single:L (project
//     mode only) takes layer L's direction for every steered layer;
//   * refused: a non-controlvector GGUF, no direction, a direction of the wrong size or type, a single:L with no
//     direction, a range that steers nothing, fewer than 2 layers.
//
// The fixtures are minimal GGUF v3 files written here (32-byte alignment, F32 tensors).
#include "../src/ds4/ds4_cvec.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {
using strata::ds4::cvec::build;

int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-70s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

template <class T> void put(std::vector<uint8_t>& b, T v) {
    const auto n = b.size();
    b.resize(n + sizeof(T));
    std::memcpy(b.data() + n, &v, sizeof(T));
}
void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put<uint64_t>(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}

struct Kv {
    std::string key;
    uint32_t type;   // 4 u32, 5 i32, 8 string
    uint64_t u = 0;
    std::string s;
};
Kv kv_u32(const std::string& k, uint32_t v) { return {k, 4, v, ""}; }
Kv kv_i32(const std::string& k, int32_t v) { return {k, 5, (uint64_t) (uint32_t) v, ""}; }
Kv kv_str(const std::string& k, const std::string& v) { return {k, 8, 0, v}; }

struct T {
    std::string name;
    std::vector<uint64_t> ne;
    std::vector<float> v;   // F32 payload (its size is the element count)
    uint32_t type = 0;
};

constexpr int kDim = 8;

std::filesystem::path write(const char* tag, const std::vector<Kv>& kvs, const std::vector<T>& ts, size_t cut = 0) {
    std::vector<uint8_t> b;
    put<uint32_t>(b, 0x46554747u);
    put<uint32_t>(b, 3);
    put<uint64_t>(b, ts.size());
    put<uint64_t>(b, kvs.size());
    for (const Kv& k : kvs) {
        put_str(b, k.key);
        put<uint32_t>(b, k.type);
        if (k.type == 4)
            put<uint32_t>(b, (uint32_t) k.u);
        else if (k.type == 5)
            put<int32_t>(b, (int32_t) (uint32_t) k.u);
        else
            put_str(b, k.s);
    }
    std::vector<uint64_t> offs;
    uint64_t off = 0;
    for (const T& t : ts) {
        put_str(b, t.name);
        put<uint32_t>(b, (uint32_t) t.ne.size());
        for (uint64_t d : t.ne) put<uint64_t>(b, d);
        put<uint32_t>(b, t.type);
        put<uint64_t>(b, off);
        offs.push_back(off);
        off += (t.v.size() * 4 + 31) / 32 * 32;
    }
    const size_t data = (b.size() + 31) / 32 * 32;
    b.resize(data + off, 0);
    for (size_t i = 0; i < ts.size(); ++i) std::memcpy(b.data() + data + offs[i], ts[i].v.data(), ts[i].v.size() * 4);
    if (cut) b.resize(b.size() - cut);
    const auto path = std::filesystem::temp_directory_path() / (std::string("strata_ds4_cvec_") + tag + ".gguf");
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(b.data()), (std::streamsize) b.size());
    return path;
}

std::vector<float> dir(float base) {
    std::vector<float> v(kDim);
    for (int i = 0; i < kDim; ++i) v[i] = base + (float) i;
    return v;
}

const std::vector<Kv> kCV = {kv_str("general.architecture", "controlvector"), kv_str("controlvector.model_hint",
                                                                                     "deepseek-v41"),
                             kv_i32("controlvector.layer_count", 4)};

// build() with the defaults the CLI leaves (first/last -1, single -1); `err` printed on a refusal
struct Out {
    std::vector<float> dir, s;
    int first = 0, last = 0;
    std::string summary, err;
};
bool run(const std::vector<std::pair<std::string, float>>& files, int n_layer, int dim, int mode, int first, int last,
         int single, Out& o) {
    const bool ok = build(files, n_layer, dim, mode, first, last, single, o.dir, o.s, o.first, o.last, o.summary, o.err);
    if (!ok) std::printf("      (%s)\n", o.err.c_str());
    return ok;
}
bool close_enough(const float* a, const float* b, int n, double tol) {
    for (int i = 0; i < n; ++i)
        if (std::fabs((double) a[i] - (double) b[i]) > tol) return false;
    return true;
}
std::vector<float> unit(const std::vector<float>& v, double nrm) {
    std::vector<float> u(v.size());
    for (size_t i = 0; i < v.size(); ++i) u[i] = (float) (v[i] / nrm);
    return u;
}
}  // namespace

int main() {
    std::vector<std::filesystem::path> made;
    auto mk = [&](std::filesystem::path p) { made.push_back(p); return p; };
    Out o;
    const int L = 5;   // the model's layers (0 .. 4)

    std::printf("add mode\n");
    {
        const auto f1 = mk(write("a1", kCV, {{"direction.2", {kDim}, dir(1.0f)}}));
        check(run({{f1.string(), 1.0f}}, L, kDim, /*add*/ 1, -1, -1, -1, o) && o.first == 1 && o.last == L - 1,
              "one file, one direction: the default range 1 .. the last layer");
        check(o.dir.size() == (size_t) L * kDim && o.s.size() == (size_t) L && o.s[(size_t) 2] == 1.0f &&
                  close_enough(o.dir.data() + 2 * kDim, dir(1.0f).data(), kDim, 0.0) &&
                  close_enough(o.dir.data(), std::vector<float>(kDim, 0.0f).data(), kDim, 0.0) &&
                  o.s[(size_t) 0] == 0.0f && o.s[(size_t) 1] == 0.0f && o.s[(size_t) 3] == 0.0f,
              "row 2 the direction, the other layers unsteered (layer 0 included)");
        check(o.summary.rfind("add:1-4", 0) == 0, "the summary names the mode and the range");
        // two files, summed with their scales
        const auto f2 = mk(write("a2", kCV, {{"direction.2", {kDim}, dir(2.0f)}}));
        std::vector<float> want(kDim);
        for (int i = 0; i < kDim; ++i) want[i] = (1.0f * (1.0f + i)) + (0.5f * (2.0f + i));
        check(run({{f1.string(), 1.0f}, {f2.string(), 0.5f}}, L, kDim, 1, -1, -1, -1, o) &&
                  close_enough(o.dir.data() + 2 * kDim, want.data(), kDim, 0.0),
              "two files: the directions times their scales, summed");
        // a steered range
        check(run({{f1.string(), 1.0f}}, L, kDim, 1, 2, 2, -1, o) && o.first == 2 && o.last == 2 && o.s[(size_t) 2] == 1.0f,
              "--control-vector-layer-range A A: one steered layer");
    }

    std::printf("project mode\n");
    {
        const auto f1 = mk(write("p1", kCV, {{"direction.2", {kDim}, dir(3.0f)}, {"direction.3", {kDim}, dir(7.0f)}}));
        std::vector<float> d = dir(3.0f);
        double nrm = 0.0;
        for (float x : d) nrm += (double) x * x;
        nrm = std::sqrt(nrm);
        check(run({{f1.string(), 1.0f}}, L, kDim, /*project*/ 0, -1, -1, -1, o), "project loads");
        const std::vector<float> u2 = unit(d, nrm);
        check(o.s[(size_t) 2] == (float) nrm && close_enough(o.dir.data() + 2 * kDim, u2.data(), kDim, 1e-6),
              "project: the unit direction, the norm as the scale");
        check(o.s[(size_t) 3] > 0.0f, "the other direction steered too");
        check(o.summary.rfind("project:1-4", 0) == 0, "the summary says project");
        // single:L: layer L's direction for every steered layer
        std::vector<float> d3 = dir(7.0f);
        double n3 = 0.0;
        for (float x : d3) n3 += (double) x * x;
        n3 = std::sqrt(n3);
        const std::vector<float> u3 = unit(d3, n3);
        check(run({{f1.string(), 1.0f}}, L, kDim, 0, 2, 3, 3, o) && o.s[(size_t) 2] == (float) n3 &&
                  close_enough(o.dir.data() + 2 * kDim, u3.data(), kDim, 1e-6) &&
                  close_enough(o.dir.data() + 3 * kDim, u3.data(), kDim, 1e-6),
              "single:3, project: layer 3's direction on layers 2 and 3");
        // single:L is project mode only: in add mode the pick is per-layer
        check(run({{f1.string(), 1.0f}}, L, kDim, 1, 2, 3, 3, o) && o.s[(size_t) 2] == 1.0f &&
                  close_enough(o.dir.data() + 2 * kDim, d.data(), kDim, 0.0),
              "single:3, add: ignored (per-layer)");
        check(!run({{f1.string(), 1.0f}}, L, kDim, 0, -1, -1, 1, o), "single:1 with no direction.1: refused");
        check(!run({{f1.string(), 1.0f}}, L, kDim, 0, -1, -1, 9, o), "single past the model: refused");
    }

    std::printf("labels\n");
    {
        const auto past = mk(write("past", kCV, {{"direction.49", {kDim}, dir(1.0f)}}));
        check(!run({{past.string(), 1.0f}}, L, kDim, 1, -1, -1, -1, o), "a direction past the model: nothing to steer");
        const auto both = mk(write("both", kCV, {{"direction.2", {kDim}, dir(1.0f)}, {"direction.49", {kDim}, dir(9.0f)}}));
        check(run({{both.string(), 1.0f}}, L, kDim, 1, -1, -1, -1, o) && close_enough(o.dir.data() + 2 * kDim,
                                                                                     dir(1.0f).data(), kDim, 0.0),
              "a past direction is ignored, the in-model one steered");
        const auto zero = mk(write("z", kCV, {{"direction.0", {kDim}, dir(1.0f)}}));
        check(!run({{zero.string(), 1.0f}}, L, kDim, 1, -1, -1, -1, o), "direction.0: layer 0 has no vector");
    }

    std::printf("refusals\n");
    {
        check(!run({}, L, kDim, 1, -1, -1, -1, o), "no files: refused");
        check(!run({{"/nonexistent/strata_ds4_cvec.gguf", 1.0f}}, L, kDim, 1, -1, -1, -1, o), "a missing file: refused");
        const auto other = mk(write("other", {kv_str("general.architecture", "llama")}, {{"direction.2", {kDim}, dir(1.0f)}}));
        check(!run({{other.string(), 1.0f}}, L, kDim, 1, -1, -1, -1, o), "a non-controlvector GGUF: refused");
        const auto none = mk(write("none", kCV, {}));
        check(!run({{none.string(), 1.0f}}, L, kDim, 1, -1, -1, -1, o), "no direction.<layer> tensors: refused");
        const auto wrong = mk(write("wrong", kCV, {{"direction.2", {kDim - 1}, dir(1.0f)}}));
        check(!run({{wrong.string(), 1.0f}}, L, kDim, 1, -1, -1, -1, o), "a direction of the wrong size: refused");
        T f16{"direction.2", {kDim}, dir(1.0f)};
        f16.type = 1;
        check(!run({{mk(write("f16", kCV, {f16})).string(), 1.0f}}, L, kDim, 1, -1, -1, -1, o),
              "a non-F32 direction: refused");
        check(!run({{mk(write("cut", kCV, {{"direction.2", {kDim}, dir(1.0f)}}, 20)).string(), 1.0f}}, L, kDim, 1, -1, -1, -1, o),
              "a file cut short inside the tensor: refused");
        check(!run({{mk(write("junk", {}, {})).string(), 1.0f}}, L, kDim, 1, -1, -1, -1, o), "not a GGUF: refused");
        check(!run({{"/nonexistent/strata_ds4_cvec.gguf", 1.0f}}, 1, kDim, 1, -1, -1, -1, o), "a one-layer model: refused");
        check(!run({{mk(write("a1", kCV, {{"direction.2", {kDim}, dir(1.0f)}})).string(), 1.0f}}, L, kDim, 1, 5, 5, -1, o),
              "a range past the model's last steerable layer: nothing to steer");
    }

    for (const auto& p : made) {
        std::error_code ec;
        std::filesystem::remove(p, ec);
    }
    std::printf("%s\n", g_fail ? "FAILED" : "all ok");
    return g_fail ? 1 : 0;
}
