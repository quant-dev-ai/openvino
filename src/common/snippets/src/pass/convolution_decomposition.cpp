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
#include "snippets/op/convolution_kernel.hpp"
#include "snippets/op/label.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/op/auto_loop.hpp"
#include "snippets/op/scalar_broadcast_load.hpp"

namespace ngraph {
namespace snippets {
namespace pass {

namespace {
void fill_body(const std::shared_ptr<Node>& node, const bool check, std::vector<std::shared_ptr<Node>>& nodes) {
    if (is_type<opset1::Result>(node) || (check && node->get_output_size() != 1ul)) {
        return;
    }

    //const auto& target_inputs = node->output(0).get_target_inputs();
    // TODO: just to test
    //if (check && target_inputs.size() != 1ul) {
    //    return;
    //}

    if (check) {
        auto &rt = node->get_rt_info();
        auto it = rt.find("LayoutDependent");
        if (it != rt.end()) {
            return;
        }
    }

    nodes.push_back(node);

    fill_body(node->output(0).get_target_inputs().begin()->get_node()->shared_from_this(), true, nodes);
}
} // namespace

namespace {
std::shared_ptr<Node> get_load(const std::shared_ptr<Node>& node) {
    if (is_type<snippets::op::Load>(node) || is_type<ngraph::opset1::Parameter>(node)) {
        return node;
    }

    if (is_type<opset1::Parameter>(node) || (node->get_input_size() != 1ul)) {
        return nullptr;
    }

    const auto parent = node->get_input_node_shared_ptr(0);
    if (parent->output(0).get_shape() != node->output(0).get_shape()) {
        return nullptr;
    }

    return get_load(parent);
}
} // namespace

// complex variant #1
//ConvolutionDecomposition::ConvolutionDecomposition() {
//    MATCHER_SCOPE(ConvolutionDecomposition);
//
//    auto matcher = ngraph::pattern::wrap_type<opset1::Convolution>();
//    ngraph::graph_rewrite_callback callback = [&](ngraph::pattern::Matcher &m) -> bool {
//        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::ConvolutionDecomposition, "Snippets::ConvolutionDecomposition")
//        auto convolution = m.get_match_root();
//        if (transformation_callback(convolution)) {
//            return false;
//        }
//
//
//
//
//
//        const auto load = get_load(convolution->get_input_node_shared_ptr(0));
//        assert(load != nullptr);
//
//        const auto load_parent = load->get_input_node_shared_ptr(0);
//        assert(load_parent->get_output_size() == 1ul);
//        const auto data_iterations_count = convolution->get_input_shape(0)[1];
//
//        const auto data_loop = std::make_shared<snippets::op::Loop>(load_parent, load_parent, data_iterations_count);
//        data_loop->set_friendly_name(convolution->get_friendly_name() + "_data_loop");
//        data_loop->get_rt_info()["order"] = 0ul;
//
//        load_parent->output(0).remove_target_input(load->input(0));
//        data_loop->input(0).replace_source_output(load_parent->output(0));
//        load->input(0).replace_source_output(data_loop->output(0));
//
//
//
//
//
//
//
//
//
//        const auto &target_inputs = convolution->output(0).get_target_inputs();
//        if (target_inputs.size() != 1ul) {
//            return false;
//        }
//
//        const auto parent = convolution->get_input_node_shared_ptr(0);
//        // TODO: NCHW
//        // TODO: static
//        const auto input_shape = convolution->get_input_shape(0);
//        const auto output_shape = convolution->output(0).get_shape();
//        // TODO: temporary assert
//        const size_t iterations_count = convolution->input(1).get_source_output().get_shape()[1];
//
//        const auto ch_loop = std::make_shared<snippets::op::Loop>(parent, parent, iterations_count);
//        ch_loop->set_friendly_name(convolution->get_friendly_name() + "_ch_loop");
//        ch_loop->get_rt_info()["order"] = 1ul;
//
//        const auto convolution_kernel = std::make_shared<snippets::op::ConvolutionKernel>(ch_loop, convolution->get_input_node_shared_ptr(1));
//        ngraph::copy_runtime_info(convolution, convolution_kernel);
//        convolution_kernel->set_friendly_name(convolution->get_friendly_name());
//
//        //const auto parent_output = parent->output(0);
//        //loop->input(0).replace_source_output(parent_output);
//        //parent_output.remove_target_input(convolution->input(0));
//
//
//        std::vector<std::shared_ptr<Node>> nodes;
//        // TODO: get the latest only
//        // TODO: return inputs (not nodes)
//        auto next = (*convolution->output(0).get_target_inputs().begin()).get_node()->shared_from_this();
//        fill_body(next, true, nodes);
//        // TODO: to debug only
//        assert(nodes.size() == 2ul);
//
//        assert(nodes.size() > 0);
//
//        auto first = nodes[0];
//        auto last = nodes.back();
//        const auto child_input = *last->output(0).get_target_inputs().begin();
//        // TODO: to debug only
//        assert(is_type<opset1::Result>(child_input.get_node()));
//
//        //first->input(0).replace_source_output(convolution_kernel->output(1));
//
//
//
//
//
//        const auto exec_jump = std::make_shared<snippets::op::ConditionalJump>(OutputVector{convolution_kernel->output(0)});
//        exec_jump->set_friendly_name(convolution->get_friendly_name() + "_exec_jump");
//        const auto exec_jump_label_begin = std::make_shared<snippets::op::Label>(OutputVector{exec_jump->output(1)});
//        exec_jump_label_begin->set_friendly_name(convolution->get_friendly_name() + "_exec_jump_begin");
//        first->input(0).replace_source_output(exec_jump_label_begin->output(0));
//        const auto exec_jump_label_end = std::make_shared<snippets::op::Label>(OutputVector{exec_jump->output(0), last});
//        exec_jump_label_end->set_friendly_name(convolution->get_friendly_name() + "_exec_jump_end");
//
//
//
//
//        const auto ch_conditional_jump = std::make_shared<snippets::op::ConditionalJump>(OutputVector{exec_jump_label_end});
//        ngraph::copy_runtime_info(convolution, ch_conditional_jump);
//        ch_conditional_jump->set_friendly_name(convolution->get_friendly_name() + "_ch_jump");
//        ch_conditional_jump->get_rt_info()["order"] = 2ul;
//        ch_loop->input(1).replace_source_output(ch_conditional_jump->output(0));
//
//        convolution->clear_control_dependents();
//        convolution->clear_control_dependencies();
//
//        //child_input.replace_source_output(ch_conditional_jump->output(1));
//
//
//        {
//            // TODO: just to check
//            assert(ch_loop->output(0).get_target_inputs().size() == 1ul);
//            assert(ch_conditional_jump->output(0).get_target_inputs().size() == 1ul);
//            //assert(ch_conditional_jump->output(1).get_target_inputs().size() == 1ul);
//            const auto expected_loop = ch_conditional_jump->output(0).get_target_inputs().begin()->get_node();
//            assert(expected_loop == ch_loop.get());
//            //const auto expected_result = ch_conditional_jump->output(1).get_target_inputs().begin()->get_node();
//            //assert(expected_result == child_input.get_node());
//        }
//
//
//
//
//
//        const auto data_conditional_jump = std::make_shared<snippets::op::ConditionalJump>(OutputVector{ch_conditional_jump->output(1)});
//        ngraph::copy_runtime_info(convolution, data_conditional_jump);
//        data_conditional_jump->set_friendly_name(convolution->get_friendly_name() + "_data_jump");
//        data_conditional_jump->get_rt_info()["order"] = 3ul;
//        data_loop->input(1).replace_source_output(data_conditional_jump->output(0));
//        child_input.replace_source_output(data_conditional_jump->output(1));
//
//        {
//            // TODO: just to check
//            assert(data_loop->output(0).get_target_inputs().size() == 1ul);
//            assert(data_conditional_jump->output(0).get_target_inputs().size() == 1ul);
//            assert(data_conditional_jump->output(1).get_target_inputs().size() == 1ul);
//            const auto expected_loop = data_conditional_jump->output(0).get_target_inputs().begin()->get_node();
//            assert(expected_loop == data_loop.get());
//            const auto expected_result = data_conditional_jump->output(1).get_target_inputs().begin()->get_node();
//            assert(expected_result == child_input.get_node());
//        }
//
//        // TODO: will be covered by tests
//        assert(convolution_kernel->output(0).get_target_inputs().size() == 1ul);
//        assert(exec_jump_label_begin->output(0).get_target_inputs().size() == 1ul);
//        assert(exec_jump_label_end->output(0).get_target_inputs().size() == 1ul);
//        return true;
//    };
//
//    register_matcher(std::make_shared<ngraph::pattern::Matcher>(matcher, matcher_name), callback);
//}

//// complex variant #2
//ConvolutionDecomposition::ConvolutionDecomposition() {
//    MATCHER_SCOPE(ConvolutionDecomposition);
//
//    auto matcher = ngraph::pattern::wrap_type<opset1::Convolution>();
//    ngraph::graph_rewrite_callback callback = [&](ngraph::pattern::Matcher &m) -> bool {
//        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::ConvolutionDecomposition, "Snippets::ConvolutionDecomposition")
//        auto convolution = m.get_match_root();
//        if (transformation_callback(convolution)) {
//            return false;
//        }
//
//        const auto biases_add = convolution->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
//        const auto biases = biases_add->get_input_node_shared_ptr(1ul);
//
//
//
//        const auto load = get_load(convolution->get_input_node_shared_ptr(0));
//        assert(load != nullptr);
//
//        //// TODO: will be fixed later: ConvolutionDecomposition will be moved upper by execution flow
//        //const auto scalar_load = std::make_shared<snippets::op::ScalarBroadcastLoad>(load->input(0).get_source_output());
//        //ngraph::copy_runtime_info(load, scalar_load);
//        //scalar_load->set_friendly_name(load->get_friendly_name());
//        //
//        //replace_node(load, scalar_load);
//        auto scalar_load = load;
//
//
//
//        const auto &target_inputs = convolution->output(0).get_target_inputs();
//        if (target_inputs.size() != 1ul) {
//            return false;
//        }
//
//        const auto parent = convolution->get_input_node_shared_ptr(0);
//        // TODO: NCHW
//        // TODO: static
//        const auto input_shape = convolution->get_input_shape(0);
//        const auto output_shape = convolution->output(0).get_shape();
//        // TODO: temporary assert
//        const size_t iterations_count = convolution->input(1).get_source_output().get_shape()[1];
//
//        const auto ch_loop = std::make_shared<snippets::op::Loop>(parent, parent, iterations_count);
//        ch_loop->set_friendly_name(convolution->get_friendly_name() + "_ch_loop");
//        ch_loop->get_rt_info()["order"] = static_cast<size_t>(1ul);
//
//        const auto convolution_kernel = std::make_shared<snippets::op::ConvolutionKernel>(
//                ch_loop,
//                convolution->get_input_node_shared_ptr(1),
//                biases);
//        ngraph::copy_runtime_info(convolution, convolution_kernel);
//        convolution_kernel->set_friendly_name(convolution->get_friendly_name());
//
//        //const auto parent_output = parent->output(0);
//        //loop->input(0).replace_source_output(parent_output);
//        //parent_output.remove_target_input(convolution->input(0));
//
//
//        std::vector<std::shared_ptr<Node>> nodes;
//        // TODO: get the latest only
//        // TODO: return inputs (not nodes)
//        auto next = biases_add->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
//        fill_body(next, true, nodes);
//        // TODO: to debug only
//        assert(nodes.size() == 2ul);
//
//        assert(nodes.size() > 0);
//
//        auto first = nodes[0];
//        auto last = nodes.back();
//        const auto child_input = *last->output(0).get_target_inputs().begin();
//        // TODO: to debug only
//        assert(is_type<opset1::Result>(child_input.get_node()));
//
//        //first->input(0).replace_source_output(convolution_kernel->output(1));
//
//
//
//        first->input(0).replace_source_output(convolution_kernel->output(0));
//
//        //const auto exec_jump = std::make_shared<snippets::op::ConditionalJump>(OutputVector{convolution_kernel->output(0)});
//        //exec_jump->set_friendly_name(convolution->get_friendly_name() + "_exec_jump");
//        //const auto exec_jump_label_begin = std::make_shared<snippets::op::Label>(OutputVector{exec_jump->output(1)});
//        //exec_jump_label_begin->set_friendly_name(convolution->get_friendly_name() + "_exec_jump_begin");
//        //first->input(0).replace_source_output(exec_jump_label_begin->output(0));
//        //const auto exec_jump_label_end = std::make_shared<snippets::op::Label>(OutputVector{exec_jump->output(0), last});
//        //exec_jump_label_end->set_friendly_name(convolution->get_friendly_name() + "_exec_jump_end");
//
//
//
//
//        //const auto ch_conditional_jump = std::make_shared<snippets::op::ConditionalJump>(OutputVector{exec_jump_label_end});
//        const auto ch_conditional_jump = std::make_shared<snippets::op::ConditionalJump>(OutputVector{last});
//        ngraph::copy_runtime_info(convolution, ch_conditional_jump);
//        ch_conditional_jump->set_friendly_name(convolution->get_friendly_name() + "_ch_jump");
//        ch_conditional_jump->get_rt_info()["order"] = static_cast<size_t>(2ul);
//        ch_loop->input(1).replace_source_output(ch_conditional_jump->output(0));
//
//        convolution->clear_control_dependents();
//        convolution->clear_control_dependencies();
//
//        //child_input.replace_source_output(ch_conditional_jump->output(1));
//
//
//        {
//            // TODO: just to check
//            assert(ch_loop->output(0).get_target_inputs().size() == 1ul);
//            assert(ch_conditional_jump->output(0).get_target_inputs().size() == 1ul);
//            //assert(ch_conditional_jump->output(1).get_target_inputs().size() == 1ul);
//            const auto expected_loop = ch_conditional_jump->output(0).get_target_inputs().begin()->get_node();
//            assert(expected_loop == ch_loop.get());
//            //const auto expected_result = ch_conditional_jump->output(1).get_target_inputs().begin()->get_node();
//            //assert(expected_result == child_input.get_node());
//        }
//
//
//
//
//
//        //const auto data_conditional_jump = std::make_shared<snippets::op::ConditionalJump>(OutputVector{ch_conditional_jump->output(1)});
//        //ngraph::copy_runtime_info(convolution, data_conditional_jump);
//        //data_conditional_jump->set_friendly_name(convolution->get_friendly_name() + "_data_jump");
//        //data_conditional_jump->get_rt_info()["order"] = 3ul;
//        //data_loop->input(1).replace_source_output(data_conditional_jump->output(0));
//        //child_input.replace_source_output(data_conditional_jump->output(1));
//        //
//        //{
//        //    // TODO: just to check
//        //    assert(data_loop->output(0).get_target_inputs().size() == 1ul);
//        //    assert(data_conditional_jump->output(0).get_target_inputs().size() == 1ul);
//        //    assert(data_conditional_jump->output(1).get_target_inputs().size() == 1ul);
//        //    const auto expected_loop = data_conditional_jump->output(0).get_target_inputs().begin()->get_node();
//        //    assert(expected_loop == data_loop.get());
//        //    const auto expected_result = data_conditional_jump->output(1).get_target_inputs().begin()->get_node();
//        //    assert(expected_result == child_input.get_node());
//        //}
//
//        child_input.replace_source_output(ch_conditional_jump->output(1));
//
//        // TODO: will be covered by tests
//        assert(convolution_kernel->output(0).get_target_inputs().size() == 1ul);
//        //assert(exec_jump_label_begin->output(0).get_target_inputs().size() == 1ul);
//        //assert(exec_jump_label_end->output(0).get_target_inputs().size() == 1ul);
//        return true;
//    };
//
//    register_matcher(std::make_shared<ngraph::pattern::Matcher>(matcher, matcher_name), callback);
//}

// complex variant #3
ConvolutionDecomposition::ConvolutionDecomposition() {
    MATCHER_SCOPE(ConvolutionDecomposition);

    auto matcher = ngraph::pattern::wrap_type<opset1::Convolution>();
    ngraph::graph_rewrite_callback callback = [&](ngraph::pattern::Matcher &m) -> bool {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::ConvolutionDecomposition, "Snippets::ConvolutionDecomposition")
        auto convolution = m.get_match_root();
        if (transformation_callback(convolution)) {
            return false;
        }

        const auto biases_add = convolution->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
        const auto biases = biases_add->get_input_node_shared_ptr(1ul);



        const auto load = get_load(convolution->get_input_node_shared_ptr(0));
        assert(load != nullptr);

        //// TODO: will be fixed later: ConvolutionDecomposition will be moved upper by execution flow
        //const auto scalar_load = std::make_shared<snippets::op::ScalarBroadcastLoad>(load->input(0).get_source_output());
        //ngraph::copy_runtime_info(load, scalar_load);
        //scalar_load->set_friendly_name(load->get_friendly_name());
        //
        //replace_node(load, scalar_load);
        auto scalar_load = load;



        const auto &target_inputs = convolution->output(0).get_target_inputs();
        if (target_inputs.size() != 1ul) {
            return false;
        }

        const auto parent = convolution->get_input_node_shared_ptr(0);
        // TODO: NCHW
        // TODO: static
        const auto input_shape = convolution->get_input_shape(0);
        const auto output_shape = convolution->output(0).get_shape();
        // TODO: temporary assert
        const size_t iterations_count = convolution->input(1).get_source_output().get_shape()[1];

        const auto loop = std::make_shared<snippets::op::Loop>(parent, parent, iterations_count);
        loop->set_friendly_name(convolution->get_friendly_name() + "_loop");
        loop->get_rt_info()["order"] = static_cast<size_t>(1ull);

        const auto convolution_kernel = std::make_shared<snippets::op::ConvolutionKernel>(
                loop->output(0),
                convolution->get_input_node_shared_ptr(1),
                biases,
                12);
        ngraph::copy_runtime_info(convolution, convolution_kernel);
        convolution_kernel->set_friendly_name(convolution->get_friendly_name());
        convolution_kernel->get_rt_info()["order"] = static_cast<size_t>(2ull);

        //const auto parent_output = parent->output(0);
        //loop->input(0).replace_source_output(parent_output);
        //parent_output.remove_target_input(convolution->input(0));


        std::vector<std::shared_ptr<Node>> nodes;
        // TODO: get the latest only
        // TODO: return inputs (not nodes)
        auto next = biases_add->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
        fill_body(next, true, nodes);
        // TODO: to debug only
        assert(nodes.size() == 2ul);

        assert(nodes.size() > 0);

        auto first = nodes[0];
        auto last = nodes.back();
        const auto return_input = *last->output(0).get_target_inputs().begin();
        // TODO: to debug only
        assert(is_type<opset1::Result>(return_input.get_node()));


        const auto auto_loop_jump = std::make_shared<snippets::op::ConditionalJump>(OutputVector{ last->output(0) });
        auto_loop_jump->set_friendly_name(convolution_kernel->get_friendly_name() + "_auto_loop_jump");
        auto_loop_jump->get_rt_info()["order"] = static_cast<size_t>(6ull);

        auto auto_loop_inputs = convolution_kernel->outputs();
        auto_loop_inputs.push_back(auto_loop_jump->output(0));
        const auto auto_loop = std::make_shared<snippets::op::AutoLoop>(auto_loop_inputs);
        auto_loop->set_friendly_name(convolution_kernel->get_friendly_name() + "_auto_loop");
        auto_loop->get_rt_info()["order"] = static_cast<size_t>(3ull);



        first->input(0).replace_source_output(auto_loop->output(0));
        first->get_rt_info()["order"] = static_cast<size_t>(4ull);


        convolution->clear_control_dependents();
        convolution->clear_control_dependencies();

        //child_input.replace_source_output(ch_conditional_jump->output(1));


        {
            // TODO: just to check
            assert(loop->output(0).get_target_inputs().size() == 1ul);
            //assert(ch_conditional_jump->output(0).get_target_inputs().size() == 1ul);
            //const auto expected_loop = ch_conditional_jump->output(0).get_target_inputs().begin()->get_node();
            //assert(expected_loop == ch_loop.get());
        }


        //return_input.replace_source_output(for_loop->output(1));
        //for_loop->input(1).replace_source_output(auto_loop_jump->output(1));

        last->get_rt_info()["order"] = static_cast<size_t>(5ull);


        const auto loop_jump = std::make_shared<snippets::op::ConditionalJump>(OutputVector{ auto_loop_jump->output(1) });
        loop_jump->set_friendly_name(convolution_kernel->get_friendly_name() + "_loop_jump");
        loop_jump->get_rt_info()["order"] = static_cast<size_t>(7ull);

        loop->input(1).replace_source_output(loop_jump->output(0));

        return_input.replace_source_output(loop_jump->output(1));

        // TODO: will be covered by tests
        assert(convolution_kernel->output(0).get_target_inputs().size() == 1ul);
        return true;
    };

    register_matcher(std::make_shared<ngraph::pattern::Matcher>(matcher, matcher_name), callback);
}

} // namespace pass
} // namespace snippets
} // namespace ngraph
