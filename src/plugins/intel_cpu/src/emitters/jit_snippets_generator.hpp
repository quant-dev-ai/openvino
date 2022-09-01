// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <cpu/x64/jit_generator.hpp>

#include "snippets/generator.hpp"

namespace ov {
namespace intel_cpu {

class jit_snippets_generator : public dnnl::impl::cpu::x64::jit_generator {
public:
    // TODO: Xbyak::LabelManager - can we use it?
    //class LabelManager {
    //public:
    //    void add(const std::string& name, const Xbyak::Label& label) {
    //        auto res = labels.emplace(name, label);
    //        if (res.second) {
    //            throw ov::Exception("label with name already exists");
    //        }
    //    }
    //    Xbyak::Label get(const std::string& name) {
    //        auto it = labels.find(name);
    //        if (it == labels.end()) {
    //            return Xbyak::Label();
    //        }
    //        return it->second;
    //    }
    //private:
    //    std::unordered_map<std::string, Xbyak::Label> labels;
    //};
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_snippet)

    ~jit_snippets_generator() = default;

    jit_snippets_generator() : jit_generator() {
    }

    void generate() override {
    }

    // TODO: Xbyak::LabelManager - can we use it?
    void L(Xbyak::Label &label, const size_t id) {
        auto res = labels.emplace(id, label);
        if (res.second) {
            throw ov::Exception("label with name already exists");
        }
        dnnl::impl::cpu::x64::jit_generator::L(label);
    }

    Xbyak::Label get_label(const size_t id) {
        auto it = labels.find(id);
        if (it == labels.end()) {
            return Xbyak::Label();
        }
        return it->second;
    }

private:
    //LabelManager label_manager;
    std::unordered_map<size_t, Xbyak::Label> labels;
};

}   // namespace intel_cpu
}   // namespace ov
