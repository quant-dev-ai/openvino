// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/convolution_kernel.hpp"
#include <assert.h>
#include "snippets/generator.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph {
namespace snippets {
namespace op {

ConvolutionKernel::ConvolutionKernel(
        const Output<Node>& data_batch,
        const Output<Node>& filters,
        const Output<Node>& biases) : Op({data_batch, filters, biases}) {
    constructor_validate_and_infer_types();
}

void ConvolutionKernel::validate_and_infer_types() {
    set_output_size(2);

    // TODO: will be implemented later
    auto input_shape = get_input_partial_shape(0);
    set_output_type(0, get_input_element_type(0), {1, 12, 112, 112, 8});
    set_output_type(1, get_input_element_type(0), {1, 12, 112, 112, 8});
}

std::shared_ptr<Node> ConvolutionKernel::clone_with_new_inputs(const OutputVector& inputs) const {
    assert(inputs.size() == 3ul);
    return std::make_shared<ConvolutionKernel>(inputs[0], inputs[1], inputs[2]);
}

} // namespace op
} // namespace snippets
} // namespace ngraph