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

ConvolutionKernel::ConvolutionKernel(const Output<Node>& parent, const Output<Node>& filters) : Op({parent}) {
}

std::shared_ptr<Node> ConvolutionKernel::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::make_shared<ConvolutionKernel>(inputs[0], inputs[1]);
}

} // namespace op
} // namespace snippets
} // namespace ngraph