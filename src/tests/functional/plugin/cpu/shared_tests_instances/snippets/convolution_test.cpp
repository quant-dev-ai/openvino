// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/convolution_test.hpp"
#include <vector>

namespace {
using namespace LayerTestsDefinitions;
using namespace ngraph;

const std::vector<ConvolutionTestValues> testValues = {
        {
                {1, 16, 112, 112},
                {{1, 1}, {0, 0}, {0, 0}, {1, 1}},
                {{1, 1}, {0, 0}, {0, 0}, {1, 1}, ov::op::PadType::EXPLICIT},
                {96, 16, 1, 1}
        },
        //{
        //        {1, 96, 112, 112},
        //        {{1, 1}, {0, 0}, {0, 0}, {1, 1}},
        //        {{1, 1}, {1, 1}, {1, 1}, {1, 1}, ov::op::PadType::EXPLICIT},
        //        {96, 1, 3, 3}
        //}
};

std::vector<size_t> input_batches = {1ul /*, 2ul*/ };

std::vector<ov::element::Type> input_types = { ov::element::f32 };

INSTANTIATE_TEST_SUITE_P(
        smoke_Snippets,
        ConvolutionTest,
        ::testing::Combine(
                ::testing::ValuesIn(testValues),
                ::testing::ValuesIn(input_batches),
                ::testing::ValuesIn(input_types),
                ::testing::Values(std::pair<size_t, size_t>{6, 0}),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ConvolutionTest::getTestCaseName);
} // namespace