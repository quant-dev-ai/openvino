// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "snippets/emitter.hpp"

namespace ngraph {
namespace snippets {
namespace op {

class Loop : public ngraph::op::Op {
public:
    OPENVINO_OP("Loop", "SnippetsOpset");

    Loop(const Output<Node>& parent, const Output<Node>& jump);
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
