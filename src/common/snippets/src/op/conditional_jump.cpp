// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/conditional_jump.hpp"
#include "snippets/generator.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph {
namespace snippets {
namespace op {

ConditionalJump::ConditionalJump(const Output<Node>& parent) : Op({parent}) {
    constructor_validate_and_infer_types();
}

void ConditionalJump::validate_and_infer_types() {
    set_output_size(2);

    const auto input_shape = get_input_partial_shape(0);
    set_output_type(0, get_input_element_type(0), input_shape);

    //auto output_shape = input_shape;
    //// NCHW8C
    //output_shape[1] = input_shape[1] * iterations_count;
    set_output_type(1, get_input_element_type(0), input_shape);

    set_input_is_relevant_to_shape(0);
}

std::shared_ptr<Node> ConditionalJump::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::make_shared<ConditionalJump>(inputs[0]/*, iterations_count*/);
}

} // namespace op
} // namespace snippets
} // namespace ngraph