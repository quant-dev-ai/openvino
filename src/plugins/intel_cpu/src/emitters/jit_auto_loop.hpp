// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>

#include "jit_snippets_emitters.hpp"
#include "jit_snippets_generator.hpp"

using namespace Xbyak;

namespace ov {
namespace intel_cpu {

class AutoLoopEmitter : public jit_emitter {
public:
    AutoLoopEmitter(jit_snippets_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n);
    size_t get_inputs_num() const override {return 2ul;}

private:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out,
                   const std::vector<size_t>& pool,
                   const std::vector<size_t>& gpr,
                   const ov::intel_cpu::emitter_context *emit_context) const override;

    template <dnnl::impl::cpu::x64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const;

private:
    size_t label_id;
    size_t iterations_count;
};
}   // namespace intel_cpu
}   // namespace ov
