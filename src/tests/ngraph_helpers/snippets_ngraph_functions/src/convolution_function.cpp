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
        const PrerequisitesParams& prerequisites_params,
        const ConvolutionParams& convolution_params,
        const ov::Shape& weights_shape) {
    assert(inputShape.size() == 4ul);
    assert(inputType == element::f32);

    const auto parameter = std::make_shared<ngraph::opset1::Parameter>(inputType, inputShape);
    parameter->set_friendly_name("parameter");

    std::shared_ptr<Node> parent = std::make_shared<ngraph::opset1::AvgPool>(
            parameter,
            prerequisites_params.strides,
            prerequisites_params.pads_begin,
            prerequisites_params.pads_end,
            prerequisites_params.kernel,
            true,
            op::RoundingType::FLOOR);
    parent->set_friendly_name("prerequisites");

    const auto generate_values = [](const Shape& shape, const float begin_value) {
        std::vector<float> values;
        values.resize(ngraph::shape_size(shape));
        for (auto i = 0; i < values.size(); ++i) {
            values[i] = begin_value + static_cast<float>(i);
        }
        return values;
    };

    const auto weights = ngraph::opset1::Constant::create(element::f32, weights_shape, generate_values(weights_shape, 10ul));
    weights->set_friendly_name("weights");

    parent = std::make_shared<ngraph::opset1::Convolution>(
        parent,
        weights,
        convolution_params.strides,
        convolution_params.pads_begin,
        convolution_params.pads_end,
        convolution_params.dilations,
        convolution_params.auto_pad);
    parent->set_friendly_name("convolution");

    const auto biases_shape = Shape{ 1, weights_shape[0ul], 1ul, 1ul };
    const auto biases = ngraph::opset1::Constant::create(element::f32, biases_shape, generate_values(biases_shape, 20ul));
    biases->set_friendly_name("biases");

    parent = std::make_shared<ngraph::opset1::Add>(parent, biases);

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
