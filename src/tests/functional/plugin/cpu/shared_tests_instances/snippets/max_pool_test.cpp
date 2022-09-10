// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/max_pool_test.hpp"
#include <vector>

using namespace LayerTestsDefinitions;
using namespace ngraph;

const std::vector<MaxPoolTestValues> testValues = {
    {
            {1, 16, 32, 32},
            {{ 1, 1 }, { 0, 0 }, { 0, 0 }, { 1, 1 }},
            {{ 4, 2 }, { 0, 0 }, { 0, 0 }, { 4, 2 }},
            {{1, 16, 1, 1}, {1, 16, 1, 1}},
    },
    {
            {1, 18, 32, 32},
            {{ 1, 1 }, { 0, 0 }, { 0, 0 }, { 1, 1 }},
            {{ 4, 2 }, { 0, 0 }, { 0, 0 }, { 4, 2 }},
            {{1, 18, 1, 1}, {1, 18, 1, 1}},
    },
    {
            {1, 64, 1024, 1024},
            {{ 1, 1 }, { 0, 0 }, { 0, 0 }, { 1, 1 }},
            {{ 4, 2 }, { 0, 0 }, { 0, 0 }, { 4, 2 }},
            {{1, 64, 1, 1}, {1, 64, 1, 1}},
    },
};

std::vector<size_t> input_batches = { 1ul /*, 2ul*/ };

std::vector<ov::element::Type> input_types = { ov::element::f32 };

INSTANTIATE_TEST_SUITE_P(
        smoke_Snippets,
        MaxPoolTest,
        ::testing::Combine(
            ::testing::ValuesIn(testValues),
            ::testing::ValuesIn(input_batches),
            ::testing::ValuesIn(input_types),
            ::testing::Values(std::pair<size_t, size_t>{4, 1}),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        MaxPoolTest::getTestCaseName);
