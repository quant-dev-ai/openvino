// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/split_load.hpp"
#include "ngraph/runtime/reference/split.hpp"

using namespace ngraph;

snippets::op::SplitLoad::SplitLoad(const Output<Node>& data, const Output<Node>& axis, const size_t num_splits) :
    Split(data, axis, num_splits) {
    //
}

//snippets::op::SplitLoad::SplitLoad(const Output<Node>& load, const Output<Node>& axis, const size_t num_splits) :
//    ngraph::op::Op({load, axis}), m_num_splits(num_splits) {
//    constructor_validate_and_infer_types();
//}
//
//void snippets::op::SplitLoad::validate_and_infer_types() {
//    std::vector<ov::PartialShape> input_shapes = {get_input_partial_shape(0), get_input_partial_shape(1)};
//    std::vector<ov::PartialShape> output_shapes;
//    shape_infer(this, input_shapes, output_shapes);
//
//    set_output_size(m_num_splits);
//    for (size_t i = 0; i < m_num_splits; ++i) {
//        set_output_type(i, get_input_element_type(0), output_shapes[i]);
//    }
//}