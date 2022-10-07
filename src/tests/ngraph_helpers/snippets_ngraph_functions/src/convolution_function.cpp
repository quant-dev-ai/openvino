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

namespace {
const auto generate_values = [](const Shape& shape, const float begin_value) {
    std::vector<float> values;
    values.resize(ngraph::shape_size(shape));
    for (auto i = 0; i < values.size(); ++i) {
        values[i] = begin_value + static_cast<float>(i);
        //values[i] = begin_value;
    }
    return values;
};

std::shared_ptr<Node> make_convolution(
        const ov::Output<ov::Node>& parent,
        const ConvolutionFunction::ConvolutionParams& convolution_params,
        const ov::Shape& weights_shape,
        const size_t index,
        const size_t size) {
    const auto weights = ngraph::opset1::Constant::create(element::f32, weights_shape, generate_values(weights_shape, 1ul));
    weights->set_friendly_name("weights" + (size == 1ul ? "" : std::to_string(index + 1)));

    const auto input_shape = parent.get_shape();

    std::shared_ptr<Node> convolution;
    if (weights_shape[1] != input_shape[1]) {
        const auto reshape = std::make_shared<ngraph::opset1::Reshape>(
            weights,
            std::make_shared<ngraph::opset1::Constant>(
                element::i64,
                Shape{ 5ull },
                std::vector<size_t>({ input_shape[1], 1ull, 1ull, weights_shape[2], weights_shape[3] })),
            true);

        convolution = std::make_shared<ngraph::opset1::GroupConvolution>(
            parent,
            reshape,
            convolution_params.strides,
            convolution_params.pads_begin,
            convolution_params.pads_end,
            convolution_params.dilations,
            convolution_params.auto_pad);
    } else {
        convolution = std::make_shared<ngraph::opset1::Convolution>(
            parent,
            weights,
            convolution_params.strides,
            convolution_params.pads_begin,
            convolution_params.pads_end,
            convolution_params.dilations,
            convolution_params.auto_pad);
    }
    convolution->set_friendly_name("convolution" + (size == 1ul ? "" : std::to_string(index + 1)));

    const auto biases_shape = Shape{ 1, weights_shape[0ul], 1ul, 1ul };
    const auto biases = ngraph::opset1::Constant::create(element::f32, biases_shape, generate_values(biases_shape, 20ul));
    biases->set_friendly_name("biases" + (size == 1ul ? "" : std::to_string(index + 1)));
    auto add = std::make_shared<ngraph::opset1::Add>(convolution, biases);
    add->set_friendly_name("add" + (size == 1ul ? "" : std::to_string(index + 1)));

    return add;
}
} // namespace

std::shared_ptr<ov::Model> ConvolutionFunction::get(
        const ngraph::Shape& inputShape,
        const element::Type inputType,
        const PrerequisitesParams& prerequisites_params,
        const std::vector<ConvolutionParams>& convolution_params) {
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

    for (auto i = 0ull; i < convolution_params.size(); ++i) {
        const auto& convolution_param = convolution_params[i];
        parent = make_convolution(parent, convolution_param, convolution_param.weights_shape, i, convolution_params.size());

        parent = std::make_shared<ngraph::opset1::Clamp>(parent, 0, 999999999999);
        parent->set_friendly_name("clamp" + (convolution_params.size() == 1ul ? "" : std::to_string(i + 1)));
    }

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
