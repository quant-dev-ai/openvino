// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "split_function.hpp"
#include "common_test_utils/data_utils.hpp"
#include <snippets/snippets_isa.hpp>
#include "function_helper.hpp"

namespace ov {
namespace test {
namespace snippets {

std::shared_ptr<ov::Model> SplitFunction::get(
    const ngraph::Shape& inputShape,
    const element::Type inputType,
    const std::vector<ngraph::Shape>& fakeQuantizeShapes,
    const float zeroPoint,
    const std::vector<std::shared_ptr<ngraph::Node>>& prerequisites,
    std::shared_ptr<ngraph::Node> operation) {
    assert(fakeQuantizeShapes.size() == 4ul);

    const auto parameter = std::make_shared<ngraph::opset1::Parameter>(inputType, inputShape);
    parameter->set_friendly_name("parameter");

    auto parent = FunctionHelper::applyPrerequisites(parameter, prerequisites);

//    const auto split_value = ngraph::opset1::Constant::create(element::f32, Shape{1, 6, 1, 1}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
//    const auto split_axis = ngraph::opset1::Constant::create(element::i64, Shape{}, { 1ul });
//    const auto split = std::make_shared<ngraph::opset1::Split>(split_value, split_axis, 2ul);
//    split->set_friendly_name("split");
//    parent = std::make_shared<ngraph::opset1::Multiply>(parent, split->outputs()[0]);
//    parent = std::make_shared<ngraph::opset1::Add>(parent, split->outputs()[1]);

    const auto multiply_value = ngraph::opset1::Constant::create(element::f32, Shape{1, 3, 1, 1}, {1.f, 2.f, 3.f});
    parent = std::make_shared<ngraph::opset1::Multiply>(parent, multiply_value);
    const auto add_value = ngraph::opset1::Constant::create(element::f32, Shape{1, 3, 1, 1}, {4.f, 5.f, 6.f});
    parent = std::make_shared<ngraph::opset1::Add>(parent, add_value);

    const auto result = std::make_shared<ngraph::opset1::Result>(parent);
    result->set_friendly_name("result");

    auto function = std::make_shared<ngraph::Function>(
            ngraph::ResultVector{ result },
            ParameterVector{ parameter },
            "SplitFunction");
    function->validate_nodes_and_infer_types();

    return function;
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
