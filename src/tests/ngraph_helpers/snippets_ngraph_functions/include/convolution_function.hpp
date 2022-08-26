// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/ngraph.hpp"

namespace ov {
namespace test {
namespace snippets {

class ConvolutionFunction {
public:
    struct PrerequisitesParams {
        ov::Strides strides;
        ov::Shape pads_begin;
        ov::Shape pads_end;
        ov::Shape kernel;
    };
    struct ConvolutionParams {
        ov::Strides strides;
        ov::CoordinateDiff pads_begin;
        ov::CoordinateDiff pads_end;
        ov::Strides dilations;
        ov::op::PadType auto_pad;
    };

    static std::shared_ptr<ov::Model> get(
            const ngraph::Shape& inputShape,
            const element::Type inputType,
            const PrerequisitesParams& prerequisites_params,
            const ConvolutionParams& convolution_params,
            const ov::Shape& weights_shape);
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
