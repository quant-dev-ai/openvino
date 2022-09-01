// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_loop.hpp"

#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/opsets/opset1.hpp>
#include "snippets/op/loop.hpp"

using namespace Xbyak;

namespace ov {
namespace intel_cpu {

LoopEmitter::LoopEmitter(
        dnnl::impl::cpu::x64::jit_generator* h,
        dnnl::impl::cpu::x64::cpu_isa_t isa,
        const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
    shouldPostIncrement = true;

    const auto& loop = as_type_ptr<ngraph::snippets::op::Loop>(n);
    iterations_count = loop->get_iterations_count();
}

void LoopEmitter::emit_impl(const std::vector<size_t>& in,
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
void LoopEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    insert_marker(MARKER_LOOP);


    //h->L(Xbyak::Label())

    insert_marker(MARKER_LOOP);
}

}   // namespace intel_cpu
}   // namespace ov
