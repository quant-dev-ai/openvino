// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>

#include "jit_snippets_emitters.hpp"

using namespace Xbyak;
using ngraph::snippets::AllocatedEmitter;

namespace ov {
namespace intel_cpu {

class ConvolutionMerged1x1KernelEmitter : public jit_emitter {
public:
    ConvolutionMerged1x1KernelEmitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const std::shared_ptr<ov::Node>& n);

    // TODO: biases?
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
    bool shouldPostIncrement;

    ov::op::PadType auto_pad;
    ov::Shape data_spatial_shape;
    ov::Shape weights_shape;

    int data_reg_index;
    int weights_reg_index;
    int biases_reg_index;
};
}   // namespace intel_cpu
}   // namespace ov
