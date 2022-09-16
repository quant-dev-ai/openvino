// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_auto_loop.hpp"

#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/opsets/opset1.hpp>
#include "snippets/op/auto_loop.hpp"

using namespace Xbyak;

namespace ov {
namespace intel_cpu {

AutoLoopEmitter::AutoLoopEmitter(
        jit_snippets_generator* h,
        dnnl::impl::cpu::x64::cpu_isa_t isa,
        const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
    const auto& auto_loop = as_type_ptr<ngraph::snippets::op::AutoLoop>(n);
    label_id = auto_loop->get_instance_id();
    input_size = auto_loop->get_input_size();
    iterations_count = auto_loop->get_input_size() - 1ull;
}

void AutoLoopEmitter::emit_impl(const std::vector<size_t>& in,
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
void AutoLoopEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    insert_marker(MARKER_LOOP);

    assert(out.size() == 1ull);

    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;

    auto h2 = static_cast<jit_snippets_generator*>(h);
    // TODO: workaround: implement and remove
    h2->uni_vmovups(Vmm(out[0]), Vmm(in[0]));

    const auto reg_index = static_cast<int>(h2->alloc_register(label_id));
    auto reg = Reg64(reg_index);
    h2->mov(reg, iterations_count);

    const auto in_size = 12ull;
    const auto vec_length = 8 * 4;

    h->sub(h->rsp, in_size * vec_length);
    for (auto i = 0ull; i < in_size; ++i) {
        push_vec(h->ptr[h->rsp + i * vec_length], in[i]);
    }


    auto label = std::make_shared<Xbyak::Label>();
    h2->L(label, label_id);

    pop_vec(out[0], h->ptr[h->rsp]);
    h->add(h->rsp, vec_length);

    insert_marker(MARKER_LOOP);
}

}   // namespace intel_cpu
}   // namespace ov
