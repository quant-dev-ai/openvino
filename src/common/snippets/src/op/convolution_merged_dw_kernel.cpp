// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/convolution_merged_dw_kernel.hpp"
#include <assert.h>
#include "snippets/generator.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph {
namespace snippets {
namespace op {

namespace {
std::vector<Output<Node>> get_inputs(
    const std::vector<Output<Node>>& data_batch,
    const Output<Node>& filters,
    const Output<Node>& biases) {
    std::vector<Output<Node>> result(data_batch);
    result.push_back(filters);
    result.push_back(biases);
    return result;
}
} // namespace

ConvolutionMergedDwKernel::ConvolutionMergedDwKernel(
        const std::vector<Output<Node>>& data_batch,
        const Output<Node>& filters,
        const Output<Node>& biases,
        const Strides& strides,
        const ov::CoordinateDiff& pads_begin,
        const ov::CoordinateDiff& pads_end,
        const Strides& dilations,
        const ov::op::PadType& auto_pad,
        const size_t outputs_size) :
        Op(get_inputs(data_batch, filters, biases)),
        strides(strides),
        pads_begin(pads_begin),
        pads_end(pads_end),
        dilations(dilations),
        auto_pad(auto_pad),
        outputs_size(outputs_size) {
    constructor_validate_and_infer_types();
}

void ConvolutionMergedDwKernel::validate_and_infer_types() {
    // TODO: just to debug: will be calculated automatically
    set_output_size(outputs_size);

    // TODO: will be implemented later
    //assert((strides.size() == 2ull) && (strides[0] == 1) && (strides[1] == 1));
    //assert((pads_begin.size() == 2ull) && (pads_begin[0] == 0) && (pads_begin[1] == 0));
    //assert((pads_end.size() == 2ull) && (pads_end[0] == 0) && (pads_end[1] == 0));
    //assert((dilations.size() == 2ull) && (dilations[0] == 1) && (dilations[1] == 1));

    auto input_shape = get_input_partial_shape(0);
    for (auto i = 0ull; i < outputs_size; ++i) {
        set_output_type(i, get_input_element_type(0), { 1, 12, 110, 110, 8 });
    }
}

//bool ConvolutionMergedDwKernel::visit_attributes(AttributeVisitor& visitor) {
//    visitor.on_attribute("strides", strides);
//    visitor.on_attribute("dilations", dilations);
//    visitor.on_attribute("pads_begin", pads_begin);
//    visitor.on_attribute("pads_end", pads_end);
//    visitor.on_attribute("auto_pad", auto_pad);
//    return true;
//}

std::shared_ptr<Node> ConvolutionMergedDwKernel::clone_with_new_inputs(const OutputVector& inputs) const {
    assert(inputs.size() >= 3ul);
    std::vector<Output<Node>> results;
    for (auto i = 0ull; i < inputs.size() - 2ull; ++i) {
        results.push_back(inputs[i]);
    }
    return std::make_shared<ConvolutionMergedDwKernel>(
        results,
        inputs[inputs.size() - 2ull],
        inputs[inputs.size() - 1ull],
        strides,
        pads_begin,
        pads_end,
        dilations,
        auto_pad,
        outputs_size);
}

ov::CoordinateDiff ConvolutionMergedDwKernel::get_pads_begin() const {
    return pads_begin;
}

ov::CoordinateDiff ConvolutionMergedDwKernel::get_pads_end() const {
    return pads_end;
}

} // namespace op
} // namespace snippets
} // namespace ngraph