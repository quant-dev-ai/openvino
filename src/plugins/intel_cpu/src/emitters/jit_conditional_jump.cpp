// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_conditional_jump.hpp"

#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/opsets/opset1.hpp>
#include "snippets/op/conditional_jump.hpp"
#include "snippets/op/label.hpp"
#include "snippets/op/auto_loop.hpp"
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
    assert(conditional_jump->output(1).get_target_inputs().size() == 1ul);

    label_ids = std::vector<size_t>(conditional_jump->get_output_size(), 0ul);

    auto loop = (*conditional_jump->output(0).get_target_inputs().begin()).get_node()->shared_from_this();

    // TODO: ILoop
    if (is_type<ngraph::snippets::op::Loop>(loop)) {
        const auto loop1 = ngraph::as_type_ptr<ngraph::snippets::op::Loop>(loop);
        iterations_count = loop1->get_iterations_count();
        label_ids[0] = loop1->get_instance_id();
    } else if (is_type<ngraph::snippets::op::AutoLoop>(loop)) {
        const auto loop2 = ngraph::as_type_ptr<ngraph::snippets::op::AutoLoop>(loop);
        iterations_count = loop2->get_iterations_count();
        label_ids[0] = loop2->get_instance_id();
    } else {
        // TODO: workaround
        iterations_count = 1ul;

        // TODO: workaround
        auto loop = as_type_ptr<ngraph::snippets::op::Loop>(conditional_jump->get_input_node_shared_ptr(0)->get_input_node_shared_ptr(0));
        assert(loop);
        register_id = loop->get_instance_id();

        // TODO: workaround
        const auto label0 = ngraph::as_type_ptr<ngraph::snippets::op::Label>(
                (*conditional_jump->output(0).get_target_inputs().begin()).get_node()->shared_from_this());
        label_ids[0] = label0->get_instance_id();
        const auto label1 = ngraph::as_type_ptr<ngraph::snippets::op::Label>(
                (*conditional_jump->output(1).get_target_inputs().begin()).get_node()->shared_from_this());
        label_ids[1] = label1->get_instance_id();
    }
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


    // TODO: hardcode - just to test
    //const auto weight_gp_1x1 = Reg64(3);
    //h->add(weight_gp_1x1, 16 * 4 * 32);
    //const auto biases_gp_1x1 = Reg64(6);
    //h->add(biases_gp_1x1, 16 * 4 * 32);

    //const auto weight_gp_dw = Reg64(8);
    //h->add(weight_gp_dw, 16 * 4 * 32);
    //const auto biases_gp_dw = Reg64(7);
    //h->add(biases_gp_dw, 16 * 4 * 32);

    auto h2 = static_cast<jit_snippets_generator*>(h);
    // TODO: workaround: implement and remove
    h2->uni_vmovups(Vmm(out[1]), Vmm(in[0]));

    if (h2->exists_label(label_ids[0])) {
        // TODO: simplify: move register management to loops
        const auto reg_index = static_cast<int>(h2->get_register(label_ids[0]));
        const auto reg = Reg64(reg_index);
        h->sub(reg, 1);
        h->cmp(reg, 1);

        const auto label = h2->get_label(label_ids[0]);
        h->jge(*label, CodeGenerator::T_NEAR);

        h2->free_register(reg_index);
    } else {
        const auto reg_index = static_cast<int>(h2->get_register(register_id));
        const auto reg = Reg64(reg_index);
        h->cmp(reg, 1);

        std::shared_ptr<Xbyak::Label> label0 = std::make_shared<Xbyak::Label>();
        h2->add_label(label_ids[0], label0);
        h->jge(*label0, CodeGenerator::T_NEAR);

        std::shared_ptr<Xbyak::Label> label1 = std::make_shared<Xbyak::Label>();
        h2->add_label(label_ids[1], label1);
        h->jl(*label1, CodeGenerator::T_NEAR);
    }

    insert_marker(MARKER_CONDITIONAL_JUMP);
}

}   // namespace intel_cpu
}   // namespace ov
