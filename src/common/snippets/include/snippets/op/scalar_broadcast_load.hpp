// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/op.hpp>
#include "load.hpp"

namespace ngraph {
namespace snippets {
namespace op {

class ScalarBroadcastLoad : public Load {
public:
    OPENVINO_OP("ScalarBroadcastLoad", "SnippetsOpset", ngraph::snippets::op::Load);

    ScalarBroadcastLoad(const Output<Node>& x);
    ScalarBroadcastLoad() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override {
        check_new_args_count(this, new_args);
        return std::make_shared<ScalarBroadcastLoad>(new_args.at(0));
    }
};

} // namespace op
} // namespace snippets
} // namespace ngraph