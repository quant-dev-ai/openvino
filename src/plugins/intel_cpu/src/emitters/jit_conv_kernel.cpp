// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_conv_kernel.hpp"

#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/opsets/opset1.hpp>
#include "snippets/op/convolution_kernel.hpp"
#include "snippets/op/loop.hpp"

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

    //const auto &convolution = as_type_ptr<ngraph::snippets::op::ConvolutionKernel>(n);
    //assert(convolution != nullptr);

    //auto get_register_index = [](const std::shared_ptr<ngraph::Node> &node) {
    //    const auto &rt = node->get_rt_info();
    //    const auto it = rt.find("reginfo");
    //    if (it == rt.end()) {
    //        throw ov::Exception("reginfo is absent");
    //    }

    //    auto regs = it->second.as<std::vector<size_t>>();
    //    if (regs.size() != 1ul) {
    //        throw ov::Exception("registers count is not correct");
    //    }
    //    return regs[0];
    //};

    //{
    //    const auto loop = convolution->get_input_node_shared_ptr(0);
    //    if (!is_type<ngraph::snippets::op::Loop>(loop)) {
    //        throw ov::Exception("unexpected operation type on data");
    //    }
    //    const auto data = loop->get_input_node_shared_ptr(0);
    //    if (!is_type<ngraph::opset1::Parameter>(data)) {
    //        throw ov::Exception("unexpected operation type on data");
    //    }
    //    data_reg_index = get_register_index(data);
    //}

    //{
    //    const auto weights = convolution->get_input_node_shared_ptr(1);
    //    if (!is_type<ngraph::opset1::Parameter>(weights)) {
    //        throw ov::Exception("unexpected operation type on weights");
    //    }
    //    weights_reg_index = get_register_index(weights);
    //    weights_shape = weights->get_shape();
    //}

    //{
    //    const auto biases = convolution->get_input_node_shared_ptr(2);
    //    if (!is_type<ngraph::opset1::Parameter>(biases)) {
    //        throw ov::Exception("unexpected operation type on biases");
    //    }
    //    biases_reg_index = get_register_index(biases);
    //}
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

namespace {
size_t get_value_offset(const size_t val_index, const size_t ch_index, const size_t filters_count, const size_t vlen) {
    //// TODO: not completed
    //if (ch_index < 8) {
    //    return val_index * 8ul * 4ul + ch_index * 4ul; // (filters_count * ch_index);
    //}

    //return val_index * 8ul * 4ul + ch_index * 4ul;

    return (val_index * 8 + (ch_index % 8) + (ch_index / 8) * 112 * 112 * 8) * 4;
}
} // namespace

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void ConvolutionKernelEmitter::emit_isa(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    assert(in.size() == 3ul);
    //assert(out.size() == 1ul);
    if (out.size() != 1ul) {
        std::cout << "ConvolutionKernelEmitter::emit_isa: why we have more outputs?" << std::endl;
    }

    insert_marker(MARKER_CONVOLUTION_KERNEL);

    int data_reg_index = in[0];
    int weights_reg_index = in[1];
    int biases_reg_index = in[2];

    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
            Xmm, isa == dnnl::impl::cpu::x64::avx2, Ymm, Zmm>::type;

    const size_t offset = dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen;
    // TODO: get from shape
    //const auto in_channels = weights_shape[1ul];
    const auto in_channels = 16ull;

    const auto data_gp = Reg64(data_reg_index);
    const auto weight_gp = Reg64(weights_reg_index);
    const auto biases_gp = Reg64(biases_reg_index);

    auto data = Vmm(15);

    std::vector<Vmm> weights = {Vmm(12), Vmm(13), Vmm(14)};
    std::vector<Vmm> accums(12ul);
    for (auto i = 0ul; i < 12ul; ++i) {
        accums[i] = Vmm(i);
    }

    // 3 * 8 = 24 channels
    for (auto ch = 0ul; ch < 3ul; ++ch) {
        // 4 values per channel
        for (auto value = 0ul; value < 4ul; ++value) {
            h->uni_vmovups(accums[0ul + 3ul * value + ch], h->ptr[biases_gp + ch * offset]);
        }
    }

    //h->uni_vmovups(weights[0ul], h->ptr[weight_gp]);
    //h->uni_vmovups(weights[1ul], h->ptr[weight_gp + 4 * in_channels * 8]);
    //h->uni_vmovups(weights[2ul], h->ptr[weight_gp + 4 * in_channels * 8 * 2]);

    const auto values_in_register = 8ul;
    // values are handled per output channel filters: 4 values per each 24 output filters
    const auto values_amount_per_channel = 4ul;
    assert(values_in_register % values_amount_per_channel == 0);

    // we need it
    //const auto data_loop = values_in_register / values_amount_per_channel;

    //h->uni_vbroadcastss(data, h->ptr[data_gp + 1024 * 1024 * 8 * 4]);

    auto v_index_begin = 0ull;
    //for (auto d_index = 0ul; d_index < data_loop; ++d_index) {
    //    // ???
    //const auto in_channels_by_reg = in_channels;
    auto ch_first = 0ull;
    for (auto filter_index = 0ull; filter_index < in_channels; ++filter_index) {
        for (auto ch = 0ull; ch < 3ull; ++ch) {
            h->uni_vmovups(weights[ch], h->ptr[weight_gp + in_channels * 8 * 4 * ch + ch_first * 8 * 4]);
        }

        // handle first values_count_per_channel_amount
        auto ch_pack = 0ull;
        for (auto index = 0ull; index < values_amount_per_channel; ++index) {
            const auto value_offset = get_value_offset(index + v_index_begin, filter_index, in_channels, offset);
            h->uni_vbroadcastss(data, h->ptr[data_gp + value_offset]);
            for (auto ch_index = 0ull; ch_index < 3ull; ++ch_index) {
                h->uni_vfmadd231ps(accums[ch_pack * 3 + ch_index], weights[ch_index], data);
            }
            ++ch_pack;
        }
        ++ch_first;
    }
    //    v_index_begin += values_amount_per_channel;
    //}

    //const auto weights_reg = Reg64(weights_reg_index);
    //Vmm weights = Vmm(in[1]);
    //h->uni_vmovups(weights, h->ptr[weights_reg]);
    //
    //Vmm output = Vmm(out[0]);
    //h->uni_vfmadd231ps(output, data, weights);
    //

    h->add(data_gp, dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen * 4ull);

    insert_marker(MARKER_CONVOLUTION_KERNEL);
}

}   // namespace intel_cpu
}   // namespace ov
