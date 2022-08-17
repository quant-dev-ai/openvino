// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/op/ops.hpp>
#include <openvino/opsets/opset1.hpp>

#include <snippets/roi_backprop/roi_backprop.hpp>
#include <openvino/pass/visualize_tree.hpp>

TEST(ROI_Backprop, TestMaxPool) {
    auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 32, 32});

    ov::Shape pad_begin{0, 0};
    ov::Shape pad_end{0, 0};
    auto max_pool = std::make_shared<ov::opset1::MaxPool>(
            input,
            ov::Strides{2, 2},
            pad_begin,
            pad_end,
            ov::Shape{2, 2},
            ov::op::RoundingType::CEIL);

    auto result = std::make_shared<ov::opset1::Result>(max_pool);

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{ input });

    ov::pass::VisualizeTree("svg/TestMaxPool.svg").run_on_model(model);

    auto map = ov::snippets::get_roi_from_function(model, {{1, 1, 8, 8}});
    auto expected_roi = std::vector<ov::PartialShape>{{1, 1, 16, 16}};
    EXPECT_EQ(map[input.get()], expected_roi);
}
