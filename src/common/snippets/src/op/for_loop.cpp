// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/for_loop.hpp"
#include "snippets/generator.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph {
namespace snippets {
namespace op {

ForLoop::ForLoop(const Output<Node>& parent, const Output<Node>& jump, const size_t iterations_count) :
        Op({ parent, jump }),
        iterations_count(iterations_count) {
    constructor_validate_and_infer_types();
}

bool ForLoop::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("iterations_count", iterations_count);
    return true;
}

void ForLoop::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<Node> ForLoop::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::make_shared<ForLoop>(inputs[0], inputs[1], iterations_count);
}

} // namespace op
} // namespace snippets
} // namespace ngraph