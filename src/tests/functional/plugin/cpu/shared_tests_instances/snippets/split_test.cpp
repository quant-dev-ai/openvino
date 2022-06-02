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
        {1, 1, 1, 1},
        {{1, 1, 1, 1}, {1, 1, 1, 1}},
        6, 1
    },
    {
        {1, 3, 1, 1},
        {{1, 3, 1, 1}, {1, 3, 1, 1}},
        6, 1
    },
    {
        {1, 3, 8, 8},
        {{1, 3, 1, 1}, {1, 3, 1, 1}},
        6, 1
    },
    {
        {1, 3, 16, 16},
        {{1, 3, 1, 1}, {1, 3, 1, 1}},
        6, 1
    },
    {
        {1, 3, 16, 16},
        {{1, 3, 16, 16}, {1, 3, 16, 16}},
        6, 1
    },
    {
        {1, 3, 16, 16},
        {{1, 3, 1, 1}, {}},
        6, 1
    },
    {
        {1, 3, 16, 16},
        {{}, {1, 3, 1, 1}},
        6, 1
    },


    {
        {1, 10, 16, 16},
        {{1, 10, 1, 1}, {1, 10, 1, 1}},
        6, 1
    },
    {
        {1, 10, 16, 16},
        {{1, 10, 16, 16}, {1, 10, 16, 16}},
        6, 1
    },
    {
        {1, 10, 16, 16},
        {{1, 10, 1, 1}, {}},
        6, 1
    },
    {
        {1, 10, 16, 16},
        {{}, {1, 10, 1, 1}},
        6, 1
    }
};

std::vector<size_t> input_batches = {
    1ul,
    2ul
};

std::vector<ov::element::Type> input_types = {
    ov::element::f32,
};

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets,
    SplitTest,
    ::testing::Combine(
        ::testing::ValuesIn(testValues),
        ::testing::ValuesIn(input_batches),
        ::testing::ValuesIn(input_types),
        ::testing::Values(std::pair<size_t, size_t>{2, 0}),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    SplitTest::getTestCaseName);

}  // namespace
