// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/label.hpp"
#include "snippets/generator.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph {
namespace snippets {
namespace op {

Label::Label(const std::vector<Output<Node>>& inputs) : Op(inputs) {
    constructor_validate_and_infer_types();
}

void Label::validate_and_infer_types() {
    set_output_size(2);

    const auto input_shape = get_input_partial_shape(0);
    set_output_type(0, get_input_element_type(0), input_shape);

    //auto output_shape = input_shape;
    //// NCHW8C
    //output_shape[1] = input_shape[1] * iterations_count;
    set_output_type(1, get_input_element_type(0), input_shape);

    set_input_is_relevant_to_shape(0);
}

std::shared_ptr<Node> Label::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::make_shared<Label>(inputs);
}

} // namespace op
} // namespace snippets
} // namespace ngraph