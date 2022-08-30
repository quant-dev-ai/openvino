// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_snippets_emitters_convolution.hpp"

#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/opsets/opset1.hpp>

using namespace Xbyak;

namespace ov {
namespace intel_cpu {

ConvolutionEmitter::ConvolutionEmitter(
        dnnl::impl::cpu::x64::jit_generator* h,
        dnnl::impl::cpu::x64::cpu_isa_t isa,
        const std::shared_ptr<ov::Node>& n) : MemoryEmitter(h, isa, n) {
    // TODO: backprop: do we need it???
    in_out_type_ = emitter_in_out_map::gpr_to_vec;
    shouldPostIncrement = true;

    const auto& max_pool = as_type_ptr<ngraph::opset1::MaxPool>(n);
    kernel = max_pool->get_kernel();
    // TODO: backprop: static shape is supported only
    input_shape = max_pool->get_input_shape(0);
}

void ConvolutionEmitter::emit_impl(const std::vector<size_t>& in,
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
void ConvolutionEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    insert_marker(MARKER_MAX_CONVOLUTION);

    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;

    Reg64 in_reg1(static_cast<int>(in[0]));
    Reg64 in_reg2(static_cast<int>(in[0]));

    Vmm vmm_in0 = Vmm(in[0]);
    Vmm vmm_out0 = Vmm(out[0]);


    h->add(in_reg1, dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen);
    h->add(in_reg2, dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen);

    insert_marker(MARKER_MAX_CONVOLUTION);
}

}   // namespace intel_cpu
}   // namespace ov
