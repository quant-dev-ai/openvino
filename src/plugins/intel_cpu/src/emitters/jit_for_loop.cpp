// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_for_loop.hpp"

#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/opsets/opset1.hpp>
#include "snippets/op/for_loop.hpp"

using namespace Xbyak;

namespace ov {
namespace intel_cpu {

ForLoopEmitter::ForLoopEmitter(
        jit_snippets_generator* h,
        dnnl::impl::cpu::x64::cpu_isa_t isa,
        const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
    const auto& loop = as_type_ptr<ngraph::snippets::op::ForLoop>(n);
    label_id = loop->get_instance_id();
    iterations_count = loop->get_iterations_count();
}

void ForLoopEmitter::emit_impl(const std::vector<size_t>& in,
                            const std::vector<size_t>& out,
                            const std::vector<size_t>& pool,
                            const std::vector<size_t>& gpr,
                            const ov::intel_cpu::emitter_context *emit_context) const {
    if (host_isa_ == dnnl::impl::cpu::x64::sse41) {
        emit_isa<dnnl::impl::cpu::x64::sse41>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx2) {
        emit_isa<dnnl::impl::cpu::x64::avx2>(in, out);
    } else if (host_isa_ == dnnl::impl::cpu::x64::avx512_common) {
        emit_isa<dnnl::impl::cpu::x64::avx512_common>(in, out);
    } else {
        IE_THROW() << host_isa_;
        assert(!"unsupported isa");
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void ForLoopEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    insert_marker(MARKER_LOOP);

    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;

    auto h2 = static_cast<jit_snippets_generator*>(h);
    // TODO: workaround: implement and remove
    h2->uni_vmovups(Vmm(out[0]), Vmm(in[0]));

    const auto reg_index = static_cast<int>(h2->alloc_register(label_id));
    auto reg = Reg64(reg_index);
    h2->mov(reg, iterations_count);
    auto label = std::make_shared<Xbyak::Label>();
    h2->L(label, label_id);

    insert_marker(MARKER_LOOP);
}

}   // namespace intel_cpu
}   // namespace ov
