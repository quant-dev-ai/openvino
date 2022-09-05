// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_label.hpp"

#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>
#include "snippets/op/conditional_jump.hpp"
#include "snippets/op/label.hpp"
#include "snippets/op/loop.hpp"

using namespace Xbyak;

namespace ov {
namespace intel_cpu {

LabelEmitter::LabelEmitter(
        jit_snippets_generator* h,
        dnnl::impl::cpu::x64::cpu_isa_t isa,
        const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
    const auto& label = as_type_ptr<ngraph::snippets::op::Label>(n);
    if (label == nullptr) {
        throw new ov::Exception("Unexpected node type");
    }

    assert(label->output(0).get_target_inputs().size() == 1ul);

    label_id = label->get_instance_id();
}

void LabelEmitter::emit_impl(const std::vector<size_t>& in,
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
void LabelEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out, const std::vector<size_t>& gpr) const {
    //assert(out.size() == 1ul);
    if (in.size() != 1ul) {
        std::cout << "LabelEmitter::emit_isa: why we have more outputs" << std::endl;
    }

    insert_marker(MARKER_LABEL);

    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;

    auto h2 = static_cast<jit_snippets_generator*>(h);
    // TODO: workaround: implement and remove
    h2->uni_vmovups(Vmm(out[0]), Vmm(in[0]));

    const auto& label = h2->get_label(label_id);
    h->L(*label);

    insert_marker(MARKER_LABEL);
}

}   // namespace intel_cpu
}   // namespace ov
