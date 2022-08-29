// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <snippets/snippets_isa.hpp>
#include "common_test_utils/data_utils.hpp"

namespace ov {
namespace test {
namespace snippets {

std::shared_ptr<ov::Model> ConvolutionFunction::get(
        const ngraph::Shape& inputShape,
        const element::Type inputType,
        const Params& prerequisites_params,
        const Params& params,
        const std::vector<ngraph::Shape>& constantShapes) {
    assert(constantShapes.size() == 2ul);

    const auto parameter = std::make_shared<ngraph::opset1::Parameter>(inputType, inputShape);
    parameter->set_friendly_name("parameter");

    const auto prerequisites = std::make_shared<ngraph::opset1::AvgPool>(
            parameter,
            prerequisites_params.strides,
            prerequisites_params.pads_begin,
            prerequisites_params.pads_end,
            prerequisites_params.kernel,
            true,
            op::RoundingType::FLOOR);
    prerequisites->set_friendly_name("prerequisites");

    const auto generate_values = [](const size_t height, const size_t width, const float begin_value) {
        std::vector<float> values;
        values.resize(height * width);
        for (auto i = 0; i < values.size(); ++i) {
            values[i] = begin_value + static_cast<float>(i);
        }
        return values;
    };

    const auto weights = ngraph::opset1::Constant::create(
            element::f32,
            Shape{ inputShape[1ul], inputShape[1ul], 1ul, 1ul },
            generate_values(inputShape[1ul], inputShape[1ul], 10ul));
    weights->set_friendly_name("weights");

    std::shared_ptr<Node> parent = std::make_shared<ngraph::opset1::Convolution>(
            prerequisites,
            weights,
            ngraph::Strides{ 1, 1 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::Strides{ 1, 1 });
    parent->set_friendly_name("convolution");

    const auto biases = ngraph::opset1::Constant::create(
            element::f32,
            Shape{ inputShape[0ul], inputShape[1ul], 1ul, 1ul },
            generate_values(inputShape[0ul], inputShape[1ul], 20ul));
    biases->set_friendly_name("biases");

    parent = std::make_shared<ngraph::opset1::Add>(parent, biases);

    //std::shared_ptr<Node> parent = prerequisites;

    //auto generate_values = [](const ngraph::Shape& shape, const float initial_value = 0.f) {
    //    std::vector<float> multiply_values(shape_size(shape));
    //    for (auto i = 0; i < multiply_values.size(); ++i) {
    //        multiply_values[i] = static_cast<float>(initial_value + i);
    //    }
    //    return multiply_values;
    //};
    //
    //const auto multiply_value = ngraph::opset1::Constant::create(
    //        element::f32,
    //        constantShapes[0],
    //        generate_values(constantShapes[0], 2.f));
    //parent = std::make_shared<ngraph::opset1::Multiply>(parent, multiply_value);
    //
    //const auto add_value = ngraph::opset1::Constant::create(
    //        element::f32,
    //        constantShapes[1],
    //        generate_values(constantShapes[1], shape_size(constantShapes[0]) + 2.f));
    //parent = std::make_shared<ngraph::opset1::Add>(parent, add_value);

    const auto result = std::make_shared<ngraph::opset1::Result>(parent);
    result->set_friendly_name("result");

    auto function = std::make_shared<ngraph::Function>(
            ngraph::ResultVector{ result },
            ParameterVector{ parameter },
            "ConvolutionFunction");
    function->validate_nodes_and_infer_types();

    return function;
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
