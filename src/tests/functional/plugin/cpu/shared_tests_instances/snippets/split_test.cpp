
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "snippets/split_test.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph;

namespace {

const std::vector<SplitTestValues> testValues = {
    {
        ov::element::f32,
        ngraph::Shape{1, 3, 16, 16},
        ov::element::f32,
        {{1, 3, 1, 1}, {1, 3, 1, 1}},
        5,
        1
    },
    {
        ov::element::f32,
        ngraph::Shape{1, 3, 16, 16},
        ov::element::f32,
        {{1, 3, 1, 1}, {}},
        5,
        1
    },
    {
        ov::element::f32,
        ngraph::Shape{1, 3, 16, 16},
        ov::element::f32,
        {{}, {1, 3, 1, 1}},
        5,
        1
    },
//    {
//        ov::element::f32,
//        ngraph::Shape{2, 3, 16, 16},
//        ov::element::f32,
//        {{1, 3, 1, 1}, {1, 3, 1, 1}},
//        5,
//        1
//    },
    {
        ov::element::f32,
        ngraph::Shape{1, 10, 16, 16},
        ov::element::f32,
        {{1, 10, 1, 1}, {1, 10, 1, 1}},
        5,
        1
    },
    {
        ov::element::f32,
        ngraph::Shape{1, 10, 16, 16},
        ov::element::f32,
        {{1, 10, 1, 1}, {}},
        5,
        1
    },
    {
        ov::element::f32,
        ngraph::Shape{1, 10, 16, 16},
        ov::element::f32,
        {{}, {1, 10, 1, 1}},
        5,
        1
    },
//    {
//        ov::element::f32,
//        ngraph::Shape{2, 10, 16, 16},
//        ov::element::f32,
//        {{1, 10, 1, 1}, {1, 10, 1, 1}},
//        5,
//        1
//    },
};

std::vector<std::pair<std::shared_ptr<Node>, std::pair<std::string, std::string>>> operations = {
    {std::make_shared<ngraph::opset1::Parameter>(), {"FakeQuantize", "fakeQuantize"}},
};

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets,
    SplitTest,
    ::testing::Combine(
        ::testing::ValuesIn(testValues),
        ::testing::ValuesIn(operations),
        ::testing::Values(std::pair<size_t, size_t>{2, 0}),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    SplitTest::getTestCaseName);

}  // namespace
