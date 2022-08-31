// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/convolution_kernel.hpp"
#include "snippets/generator.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph {
namespace snippets {
namespace op {

ConvolutionKernel::ConvolutionKernel(const Output<Node>& parent, const Output<Node>& filters) : Op({parent, filters}) {
    constructor_validate_and_infer_types();
}

void ConvolutionKernel::validate_and_infer_types() {
    auto input_shape = get_input_partial_shape(0);
    set_output_type(0, get_input_element_type(0), {1, 12, 112, 112, 8});
}

std::shared_ptr<Node> ConvolutionKernel::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::make_shared<ConvolutionKernel>(inputs[0], inputs[1]);
}

} // namespace op
} // namespace snippets
} // namespace ngraph