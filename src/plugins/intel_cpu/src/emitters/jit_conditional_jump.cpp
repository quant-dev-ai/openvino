// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_conditional_jump.hpp"

#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/opsets/opset1.hpp>
#include "snippets/op/conditional_jump.hpp"
#include "snippets/op/loop.hpp"

using namespace Xbyak;

namespace ov {
namespace intel_cpu {

ConditionalJumpEmitter::ConditionalJumpEmitter(
        jit_snippets_generator* h,
        dnnl::impl::cpu::x64::cpu_isa_t isa,
        const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
    const auto& conditional_jump = as_type_ptr<ngraph::snippets::op::ConditionalJump>(n);
    //iterations_count = conditional_jump->get_iterations_count();
    assert(conditional_jump->output(0).get_target_inputs().size() == 1ul);

    const auto loop = ngraph::as_type_ptr<ngraph::snippets::op::Loop>(
            (*conditional_jump->output(0).get_target_inputs().begin()).get_node()->shared_from_this());
    assert(loop != nullptr);
    iterations_count = loop->get_iterations_count();
    label_id = loop->get_instance_id();
}

void ConditionalJumpEmitter::emit_impl(const std::vector<size_t>& in,
                            const std::vector<size_t>& out,
                            const std::vector<size_t>& pool,
                            const std::vector<size_t>& gpr,
                            const ov::intel_cpu::emitter_context *emit_context) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out, gpr);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out, gpr);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_common) {
        emit_isa<dnnl::impl::cpu::x64::avx512_common>(in, out, gpr);
    } else {
        IE_THROW() << host_isa_;
        assert(!"unsupported isa");
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void ConditionalJumpEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out, const std::vector<size_t>& gpr) const {
    //assert(in.size() == 1ul);
    if (in.size() != 1ul) {
        std::cout << "ConditionalJumpEmitter::emit_isa: why I have two inputs" << std::endl;
    }
    // TODO: we need only one output register
    assert(out.size() == 2ul);

    insert_marker(MARKER_CONDITIONAL_JUMP);

    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;

    auto h2 = static_cast<jit_snippets_generator*>(h);
    // TODO: workaround: implement and remove
    h2->uni_vmovups(Vmm(out[1]), Vmm(in[0]));

    const auto reg_index = static_cast<int>(h2->get_register(label_id));
    const auto reg = Reg64(reg_index);
    h->sub(reg, 1);
    h->cmp(reg, 1);

    const auto label = h2->get_label(label_id);
    h->jge(label, CodeGenerator::T_NEAR);

    h2->free_register(reg_index);

    insert_marker(MARKER_CONDITIONAL_JUMP);
}

}   // namespace intel_cpu
}   // namespace ov
