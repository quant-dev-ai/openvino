// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "snippets/emitter.hpp"

namespace ngraph {
namespace snippets {
namespace op {

class AutoLoop : public ngraph::op::Op {
public:
    OPENVINO_OP("AutoLoop", "SnippetsOpset");

    AutoLoop(const Output<Node>& parent, const Output<Node>& jump, const size_t iterations_count);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

    size_t get_iterations_count() const { return iterations_count; }

private:
    // TODO: unsigned short?
    size_t iterations_count;
};

} // namespace op
} // namespace snippets
} // namespace ngraph
