// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "snippets/emitter.hpp"
#include "tile.hpp"

namespace ngraph {
namespace snippets {
namespace op {

class ConditionalJump : public ngraph::op::Op {
public:
    OPENVINO_OP("ConditionalJump", "SnippetsOpset");

    // TODO: one parent only
    ConditionalJump(const std::vector<Output<Node>>& inputs);

    bool visit_attributes(AttributeVisitor& visitor) override { return true; }

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;
};

} // namespace op
} // namespace snippets
} // namespace ngraph
