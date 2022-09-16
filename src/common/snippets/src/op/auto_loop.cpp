// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/auto_loop.hpp"
#include "snippets/generator.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph {
namespace snippets {
namespace op {

AutoLoop::AutoLoop(const OutputVector& arguments) : Op(arguments), iterations_count(arguments.size() - 1ull) {
    constructor_validate_and_infer_types();
}

bool AutoLoop::visit_attributes(AttributeVisitor& visitor) {
    return true;
}

void AutoLoop::validate_and_infer_types() {
    set_output_size(1);
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<Node> AutoLoop::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::make_shared<AutoLoop>(inputs);
}

} // namespace op
} // namespace snippets
} // namespace ngraph