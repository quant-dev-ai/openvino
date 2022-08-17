// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/gather.hpp>
#include <openvino/core/partial_shape.hpp>

namespace ov {
namespace op {
namespace v1 {
template <class ShapeType>
void roi_backprop(const MaxPool* op, const std::vector<ShapeType>& input_shapes, std::vector<ShapeType>& roi_shapes) {
    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 1ul) && (roi_shapes.size() == 1));
    NODE_VALIDATION_CHECK(op, input_shapes[0].rank() == roi_shapes[0].rank());


    const auto kernel = op->get_kernel();
    //NODE_VALIDATION_CHECK(op, input_shapes[0].rank() == (kernel.size() + 2ul));

    // TODO: roi backpropagation: dynamic shape can be used
    // TODO: roi backpropagation: params are ignored

    //const auto& input_shape = input_shapes[0].get_max_shape();
    //Shape roi_shape = input_shape;
    //roi_shape[0ul] = 1ul;
    //roi_shape[1ul] = 1ul;
    //
    //if (input_shape.size() == 3ul) {
    //    assert((input_shape[2ul] % kernel[0]) == 0);
    //    roi_shape[2ul] = input_shape[2ul] / kernel[0];
    //} else {
    //    for (auto i = 2ul; i < input_shape.size(); ++i) {
    //        assert((input_shape[i] % kernel[i - 2ul]) == 0);
    //        roi_shape[i] = input_shape[i] / kernel[i - 2ul];
    //    }
    //}

    auto roi_shape = roi_shapes[0];

    // TODO: backprop: looks like as workaround - add comment
    auto shape_offset = input_shapes[0].size() - (kernel.size() + 2ul);
    for (auto i = 2ul; i < roi_shape.size() - shape_offset; ++i) {
        roi_shape[i] = roi_shape[i] * kernel[i - 2ul];
    }

    roi_shapes[0] = roi_shape;
}

}  // namespace v1
}  // namespace op
}  // namespace ov