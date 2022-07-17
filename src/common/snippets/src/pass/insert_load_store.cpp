// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>
#include "snippets/remarks.hpp"

#include "snippets/pass/insert_load_store.hpp"
#include "snippets/snippets_isa.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

ngraph::snippets::pass::InsertLoad::InsertLoad() {
    MATCHER_SCOPE(InsertLoad);
    register_matcher(std::make_shared<ngraph::pattern::Matcher>(
        ngraph::pattern::wrap_type<ngraph::opset1::Parameter>()),
            [this](ngraph::pattern::Matcher &m) {
            OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::InsertLoad")
            auto root = m.get_match_root();

            // TODO: PoC
            bool handled = false;
#ifdef SNIPPETS_DEBUG
            std::cout << "root: " << root->get_friendly_name() << std::endl;
#endif
            if ((root->outputs().size() == 1ul) && (root->output(0).get_target_inputs().size() == 1ul)) {
                for (auto& root_output : root->outputs()) {
                    for (auto& root_output_input : root_output.get_target_inputs()) {
                        auto split = ngraph::as_type_ptr<opset1::Split>(root_output_input.get_node()->shared_from_this());
                        if (split != nullptr) {
                            std::cout << "split: " << split->get_friendly_name() << std::endl;
                            auto offset = 0ul;
                            const auto output_size = split->get_output_size();
                            for (auto split_output_index = 0ul; split_output_index < output_size; ++split_output_index) {
                                const auto& split_output = split->output(split_output_index);
                                // TODO: need test for that
                                for (auto& split_output_input : split_output.get_target_inputs()) {
                                    //auto split_consumer = split_input.get_node()->shared_from_this();

                                    //auto load = std::make_shared<ngraph::snippets::op::Load>(root_output);
                                    auto load = std::make_shared<ngraph::snippets::op::Load>(split_output);
                                    split_output_input.replace_source_output(load->output(0));
#ifdef SNIPPETS_DEBUG
                                    std::cout << "split consumer: " << split_output_input.get_node()->get_friendly_name() << std::endl;
#endif

                                    ngraph::copy_runtime_info(root, load);

                                    // TODO: fake split consumers order is important: the latest Load increment memory pointer
                                    // TODO: fake split: add check: has to have only one consumer for one output port
//                                    auto friendly_name = load->get_friendly_name();
//                                    auto index = split_output.get_index();
//                                    auto output_size = split->get_output_size() - 1;
//                                    std::cout << "friendly_name=" << friendly_name <<
//                                        ", index=" << index <<
//                                        ", output_size=" << output_size << std::endl << std::endl;

//                                    if (friendly_name == "Load_2900") {
//                                        std::cout << "DEBUG: please fix me: remove hardcode" << std::endl;
//                                        load->offset = 8ul * 4ul * 2ul;
//                                    }

//                                    if (index == output_size) {
//                                        std::cout << "DEBUG: please fix me: remove hardcode" << std::endl;
//                                        load->offset = 8ul * 4ul * 2ul;
//                                    }

                                    if (split_output_index != 0) {
                                        load->offset = offset;
                                    }

//                                    const auto shape = split_output.get_shape();
//                                    // TODO: concatenation was done by batch <= add tests to ignore: when constant has batch
//                                    const auto length_in_byte = split_output.get_element_type().bitwidth() / 8;
                                    offset += ov::shape_size(split_output.get_shape()) * split_output.get_element_type().bitwidth() / 8;
                                    handled = true;
                                }
                            }
                        }
                    }
                }
            }

            if (handled) {
                return true;
            }

            // check if already has Load as an output
            for (auto output : root->outputs()) {
                for (auto consumer : output.get_target_inputs()) {
                    if (ov::is_type<ngraph::snippets::op::Load>(consumer.get_node())) {
                        return false;
                    }
                }
            }

            auto load = std::make_shared<ngraph::snippets::op::Load> (root);
            ngraph::copy_runtime_info(root, load);

            bool rewritten = false;
            for (auto output : root->outputs()) {
                for (auto consumer : output.get_target_inputs()) {
                    if (consumer.get_node()->shared_from_this() != load) {
                        consumer.replace_source_output(load);
                        rewritten |= true;
                    }
                }
            }

            return rewritten;
        });
}

ngraph::snippets::pass::InsertStore::InsertStore() {
    MATCHER_SCOPE(InsertStore);
    register_matcher(std::make_shared<ngraph::pattern::Matcher>(
        ngraph::pattern::wrap_type<ngraph::opset1::Result>()),
            [this](ngraph::pattern::Matcher &m) {
            OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::InsertStore")
            auto root = m.get_match_root();

            // check if already has Store as an input
            for (auto input : root->inputs()) {
                if (ov::is_type<ngraph::snippets::op::Store>(input.get_source_output().get_node())) {
                    return false;
                }
            }

            auto store = std::make_shared<ngraph::snippets::op::Store> (root->input_value(0));
            ngraph::copy_runtime_info(root, store);
            root->set_argument(0, store);
            return true;
        });
}
