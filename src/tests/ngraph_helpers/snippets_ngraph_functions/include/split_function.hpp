// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/ngraph.hpp"

namespace ov {
namespace test {
namespace snippets {

class SplitFunction {
public:
    static std::shared_ptr<ov::Model> get(
        const ngraph::Shape& inputShape,
        const element::Type inputType,
        const std::vector<ngraph::Shape>& fakeQuantizeShapes,
        const float zeroPoint,
        const std::vector<std::shared_ptr<ngraph::Node>>& prerequisites,
        std::shared_ptr<ngraph::Node> operation = nullptr);
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
