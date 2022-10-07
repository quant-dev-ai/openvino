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
        const ov::CoordinateDiff& pads_begin,
        const ov::CoordinateDiff& pads_end,
        const size_t outputs_size) : Op(get_inputs(data_batch, filters, biases)), pads_begin(pads_begin), pads_end(pads_end), outputs_size(outputs_size) {
    constructor_validate_and_infer_types();
}

void ConvolutionMergedDwKernel::validate_and_infer_types() {
    // TODO: just to debug: will be calculated automatically
    set_output_size(outputs_size);

    // TODO: will be implemented later
    auto input_shape = get_input_partial_shape(0);
    for (auto i = 0ull; i < outputs_size; ++i) {
        set_output_type(i, get_input_element_type(0), { 1, 12, 112, 112, 8 });
    }
}

std::shared_ptr<Node> ConvolutionMergedDwKernel::clone_with_new_inputs(const OutputVector& inputs) const {
    assert(inputs.size() >= 3ul);
    std::vector<Output<Node>> results;
    for (auto i = 0ull; i < inputs.size() - 2ull; ++i) {
        results.push_back(inputs[i]);
    }
    return std::make_shared<ConvolutionMergedDwKernel>(results, inputs[inputs.size() - 2ull], inputs[inputs.size() - 1ull], pads_begin, pads_end, outputs_size);
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