// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "snippets/pass/replace_loads_with_split_loads.hpp"
#include "snippets/snippets_isa.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <snippets/op/split_load.hpp>

ngraph::snippets::pass::ReplaceLoadsWithSplitLoads::ReplaceLoadsWithSplitLoads() {
    MATCHER_SCOPE(ReplaceLoadsWithSplitScalarLoads);

    auto load_wrapper = ngraph::pattern::wrap_type<ngraph::snippets::op::Load>();
    auto constant_wrapper = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    auto split_wrapper = ngraph::pattern::wrap_type<ngraph::opset1::Split>({load_wrapper, constant_wrapper});
    auto matcher = std::make_shared<ngraph::pattern::Matcher>(split_wrapper);
    auto callback = [this](ngraph::pattern::Matcher &m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::ReplaceLoadsWithSplitScalarLoads")
        auto current_split = m.get_match_root();
        if (transformation_callback(current_split))
            return false;

        auto parent1 = current_split->get_input_node_shared_ptr(0)->get_input_node_shared_ptr(0);
        auto parent2 = current_split->get_input_node_shared_ptr(1);
        auto new_split = std::make_shared<ngraph::snippets::op::SplitLoad>(
                parent1, // TODO: refactor
                parent2,
                current_split->get_output_size());

        new_split->set_friendly_name(current_split->get_friendly_name());
        ngraph::copy_runtime_info(current_split, new_split);

        ngraph::replace_node(current_split, new_split);
        return true;
    };

    register_matcher(matcher, callback);
}
