// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_conv_kernel.hpp"

#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/opsets/opset1.hpp>
#include "snippets/op/convolution_kernel.hpp"

using namespace Xbyak;

namespace ov {
namespace intel_cpu {

ConvolutionKernelEmitter::ConvolutionKernelEmitter(
        dnnl::impl::cpu::x64::jit_generator* h,
        dnnl::impl::cpu::x64::cpu_isa_t isa,
        const std::shared_ptr<ov::Node>& n) : jit_emitter(h, isa, n) {
    // TODO: backprop: do we need it???
    in_out_type_ = emitter_in_out_map::gpr_to_vec;
    shouldPostIncrement = true;

    const auto& convolution = as_type_ptr<ngraph::snippets::op::ConvolutionKernel>(n);
    assert(convolution != nullptr);

    const auto load = convolution->get_input_node_shared_ptr(1);
    if (!is_type<ngraph::snippets::op::Load>(load)) {
        throw ov::Exception("unexpected operation type on weights");
    }

    const auto weights = load->get_input_node_shared_ptr(0);
    if (!is_type<ngraph::opset1::Parameter>(weights)) {
        throw ov::Exception("unexpected operation type on weights");
    }

    weights_shape = weights->get_shape();

    const auto& rt = weights->get_rt_info();
    const auto it = rt.find("reginfo");
    if (it == rt.end()) {
        throw ov::Exception("reginfo is absent");
    }

    auto regs = it->second.as<std::vector<size_t>>();
    if (regs.size() != 1ul) {
        throw ov::Exception("registers count is not correct");
    }
    weights_reg_index = regs[0];
}

void ConvolutionKernelEmitter::emit_impl(const std::vector<size_t>& in,
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
void ConvolutionKernelEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    assert(in.size() == 2ul);
    assert(out.size() == 1ul);

    insert_marker(MARKER_CONVOLUTION_KERNEL);

    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;

    //const auto offset = dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen;

    Vmm data = Vmm(in[0]);
    const auto weights_reg = Reg64(weights_reg_index);
    Vmm weights = Vmm(in[1]);
    h->uni_vmovups(weights, h->ptr[weights_reg]);
    // TODO: just to debug
    h->uni_vmovups(weights, h->ptr[weights_reg + 8 * 4]);
    h->uni_vmovups(weights, h->ptr[weights_reg + 8 * 4 * 2]);

    Vmm output = Vmm(out[0]);
    h->uni_vfmadd231ps(output, data, weights);

    insert_marker(MARKER_CONVOLUTION_KERNEL);
}

}   // namespace intel_cpu
}   // namespace ov
