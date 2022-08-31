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

ConditionalJump::ConditionalJump(const Output<Node>& parent, const Output<Node>& loop) : Op({parent}) {
}

std::shared_ptr<Node> ConditionalJump::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::make_shared<ConditionalJump>(inputs[0], inputs[1]);
}

} // namespace op
} // namespace snippets
} // namespace ngraph