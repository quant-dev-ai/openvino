
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/split_test.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>
#include "ngraph_ops/type_relaxed.hpp"
#include "split_function.hpp"
#include "function_helper.hpp"

namespace LayerTestsDefinitions {

std::string SplitTest::getTestCaseName(testing::TestParamInfo<testsParams> obj) {
    std::ostringstream result;
    const auto values = std::get<0>(obj.param);
    const auto operation = std::get<1>(obj.param);
    const auto operations_number = std::get<2>(obj.param);
    const auto targetDevice = std::get<3>(obj.param);

    const auto type_info = operation.first->get_type_info();
    const auto operationString = ngraph::is_type<ngraph::opset1::Parameter>(operation.first) ?
        "nullptr" :
        (std::string(type_info.name) + "_" + std::string(type_info.version_id));

    result << "IS=" << CommonTestUtils::vec2str(values.inputShape) << "_";
    result << "netPRC=" << values.modelType << "_";
    result << "D=" << targetDevice << "_";
    result << "IN=" << values.inputType << "_";
    result << "OP=" << operationString << "_";
    result << "ON1=" << std::string(operation.second.first) << "_";
    result << "ON1=" << std::string(operation.second.second) << "_";
    result << "NN=" << values.num_nodes;
    for (auto i = 0; i < values.constantShapes.size(); ++i) {
        result << "_SH" << i << "=" << values.constantShapes[i];
    }
    return result.str();
}

void SplitTest::SetUp() {
    // TODO: why is default abs_threshold value 385.25 ???
    abs_threshold = 0.01;
    // TODO: why is default rel_threshold value 1.79769e+308
    rel_threshold = 0.01;

    auto& testsParams = this->GetParam();

    const auto values = std::get<0>(testsParams);
    const auto operation = std::get<1>(testsParams);
    const auto operations_number = std::get<2>(testsParams);
    targetDevice = std::get<3>(testsParams);

    ref_num_nodes = operations_number.first;
    ref_num_subgraphs = operations_number.second;

    init_input_shapes({{values.inputShape, {values.inputShape}}});

    std::shared_ptr<ngraph::Node> op = ngraph::is_type<ngraph::opset1::Parameter>(operation.first) ? nullptr : operation.first;
    function = ov::test::snippets::SplitFunction::get(
        {values.inputShape},
        values.inputType,
        values.constantShapes,
        ov::test::snippets::FunctionHelper::makePrerequisitesOriginal(),
        op);

    ngraph::pass::VisualizeTree("svg/test.actual.svg").run_on_model(function);
}

void SplitTest::run() {
    SubgraphBaseTest::run();

    const auto params = std::get<0>(GetParam());
    this->ref_num_nodes = params.num_nodes;
    this->ref_num_subgraphs = params.num_subgraphs;
    validateNumSubgraphs();
}

TEST_P(SplitTest, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
