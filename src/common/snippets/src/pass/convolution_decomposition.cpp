// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/convolution_decomposition.hpp"

#include "snippets/remarks.hpp"
#include <snippets/itt.hpp>
#include "snippets/op/subgraph.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/rt_info.hpp>
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>
#include <cassert>
#include <queue>
#include <string>
#include <numeric>
#include <climits>

#include "snippets/op/conditional_jump.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/op/convolution_kernel.hpp"

namespace ngraph {
namespace snippets {
namespace pass {

ConvolutionDecomposition::ConvolutionDecomposition() {
    MATCHER_SCOPE(ConvolutionDecomposition);

    auto matcher = ngraph::pattern::wrap_type<opset1::Convolution>();
    ngraph::graph_rewrite_callback callback = [&](ngraph::pattern::Matcher &m) -> bool {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::ConvolutionDecomposition, "Snippets::ConvolutionDecomposition")
        auto convolution = m.get_match_root();
        if (transformation_callback(convolution)) {
            return false;
        }

        const auto& target_inputs = convolution->output(0).get_target_inputs();
        if (target_inputs.size() != 1ul) {
            return false;
        }

        const auto loop = std::make_shared<snippets::op::Loop>(convolution->get_input_node_shared_ptr(0)->output(0));
        ngraph::copy_runtime_info(convolution, loop);
        loop->set_friendly_name(convolution->get_friendly_name() + "_loop");

        const auto convolution_kernel = std::make_shared<snippets::op::ConvolutionKernel>(loop, convolution->get_input_node_shared_ptr(1));
        ngraph::copy_runtime_info(convolution, convolution_kernel);
        convolution_kernel->set_friendly_name(convolution->get_friendly_name());

        const auto conditional_jump = std::make_shared<snippets::op::ConditionalJump>(convolution_kernel, loop);
        ngraph::copy_runtime_info(convolution, conditional_jump);
        conditional_jump->set_friendly_name(convolution->get_friendly_name() + "_jump");


        ngraph::replace_node(convolution, conditional_jump);

        return true;
    };

    register_matcher(std::make_shared<ngraph::pattern::Matcher>(matcher, matcher_name), callback);
}

} // namespace pass
} // namespace snippets
} // namespace ngraph
