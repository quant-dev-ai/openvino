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

class ConvolutionKernel : public ngraph::op::Op {
public:
    OPENVINO_OP("ConvolutionKernel", "SnippetsOpset");

    ConvolutionKernel(const Output<Node>& parent, const Output<Node>& filters);
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;
};

} // namespace op
} // namespace snippets
} // namespace ngraph
