// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/convolution_merged_1x1_kernel.hpp"
#include <assert.h>
#include "snippets/generator.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph {
namespace snippets {
namespace op {

ConvolutionMerged1x1Kernel::ConvolutionMerged1x1Kernel(
        const Output<Node>& data_batch,
        const Output<Node>& filters,
        const Output<Node>& biases,
        const size_t outputs_size) : Op({data_batch, filters, biases}), outputs_size(outputs_size) {
    constructor_validate_and_infer_types();
}

void ConvolutionMerged1x1Kernel::validate_and_infer_types() {
    // TODO: just to debug: will be calculated automatically
    set_output_size(outputs_size);

    // TODO: will be implemented later
    auto input_shape = get_input_partial_shape(0);
    for (auto i = 0ull; i < outputs_size; ++i) {
        if (input_shape == PartialShape{1, 2, 112, 112, 8}) {
            set_output_type(i, get_input_element_type(0), {1, 12, 112, 112, 8});
        } else if (input_shape == PartialShape{1, 2, 224, 224, 8}) {
            set_output_type(i, get_input_element_type(0), {1, 12, 224, 224, 8});
        } else {
            throw ov::Exception("not expected input shape");
        }
    }
}

std::shared_ptr<Node> ConvolutionMerged1x1Kernel::clone_with_new_inputs(const OutputVector& inputs) const {
    assert(inputs.size() == 3ul);
    return std::make_shared<ConvolutionMerged1x1Kernel>(inputs[0], inputs[1], inputs[2], outputs_size);
}

} // namespace op
} // namespace snippets
} // namespace ngraph