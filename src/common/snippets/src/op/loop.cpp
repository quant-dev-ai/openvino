// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/loop.hpp"
#include "snippets/generator.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph {
namespace snippets {
namespace op {

Loop::Loop(const Output<Node>& parent) : Op({ parent }) {
    constructor_validate_and_infer_types();
}

void Loop::validate_and_infer_types() {
    set_output_size(2);

    std::vector<ov::PartialShape> input_shapes = { get_input_partial_shape(0) };
    set_output_type(0, get_input_element_type(0), input_shapes[0]);
    set_output_type(0, get_input_element_type(0), input_shapes[0]);

    set_input_is_relevant_to_shape(0);
}

std::shared_ptr<Node> Loop::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::make_shared<Loop>(inputs[0]);
}

} // namespace op
} // namespace snippets
} // namespace ngraph