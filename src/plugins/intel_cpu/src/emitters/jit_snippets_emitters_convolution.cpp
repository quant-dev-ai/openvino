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

    const auto& convolution = as_type_ptr<ngraph::opset1::Convolution>(n);
    auto_pad = convolution->get_auto_pad();

    weights_shape = convolution->get_input_node_shared_ptr(1)->get_shape();
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

    //const auto offset = dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen;

    Reg64 in_reg1(static_cast<int>(in[0]));
    Reg64 in_reg2(static_cast<int>(in[1]));

    Vmm data = Vmm(0);
    h->uni_vmovups(data, h->ptr[in_reg1]);

    Vmm weights = Vmm(1);
    h->uni_vmovups(weights, h->ptr[in_reg2]);

    Vmm output = Vmm(3);
    h->uni_vfmadd231ps(output, data, weights);

    //Vmm vmm1 = Vmm(1);
    //h->uni_vmovups(vmm1, h->ptr[in_reg1 + offset * weights_shape]);

    //Vmm vmm2 = Vmm(2);
    //h->uni_vmovups(vmm1, h->ptr[in_reg1 + offset]);


    h->add(in_reg1, dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen);
    h->add(in_reg2, dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen);

    insert_marker(MARKER_MAX_CONVOLUTION);
}

}   // namespace intel_cpu
}   // namespace ov
