// src/glm46/glm46_env.hpp - maya-glm46's environment settings: MAYA_<name> first (the other engines' prefix), then <name>
// itself (the GLM46_* names #477 used, kept as aliases).
#pragma once

#include <cstdlib>
#include <string>

namespace glm46 {

inline const char* env(const char* name) {
    const std::string m = std::string("MAYA_") + name;
    if (const char* v = std::getenv(m.c_str())) return v;
    return std::getenv(name);
}

}  // namespace glm46
