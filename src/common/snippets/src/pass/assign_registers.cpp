// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// #include <openvino/cc/selective_build.h>
#include <snippets/itt.hpp>
#include "snippets/remarks.hpp"

#include "snippets/pass/assign_registers.hpp"
#include "snippets/snippets_isa.hpp"

#include <ngraph/opsets/opset1.hpp>

#include <iterator>
#include <openvino/runtime/exception.hpp>

#include <ngraph/pass/visualize_tree.hpp>

namespace {
const std::vector<size_t> get_registers(const std::shared_ptr<ngraph::Node>& node) {
    auto &rt = node->get_rt_info();
    auto it = rt.find("reginfo");
    if (it == rt.end()) {
        return {};
    }

    auto &registers = it->second.as<std::vector<size_t>>();
    return registers;
}

size_t get_max_register(const std::shared_ptr<ov::Model>& f) {
    auto max_register = 0ul;
    for (const auto& n : f->get_ordered_ops()) {
        auto registers = get_registers(n);
        for (const auto reg : registers) {
            if (max_register < reg) {
                max_register = reg;
            }
        }
    }
    return max_register;
}

size_t get_max_constant_register(const std::shared_ptr<ov::Model>& f) {
    auto max_register = 0ul;
    for (const auto& n : f->get_ordered_ops()) {
        if (!ngraph::is_type<ngraph::opset1::Constant>(n)) {
            continue;
        }

        auto registers = get_registers(n);
        for (const auto reg : registers) {
            if (max_register < reg) {
                max_register = reg;
            }
        }
    }
    return max_register;
}

void fix_concatenation(const std::shared_ptr<ngraph::Node>& node, const size_t register_number) {
    auto& rt = node->get_rt_info();
    auto it = rt.find("reginfo");
    if (it == rt.end()) {
        throw ov::Exception("register is absent");
    }

    auto& registers = it->second.as<std::vector<size_t>>();
    if (registers.size() != 1ul) {
        throw ov::Exception("unexpected registers count");
    }

    registers[0] = register_number;
}

void fix_concatenation_up(const std::shared_ptr<ngraph::Node>& node, const size_t register_number) {
    if (!ngraph::is_type<ngraph::snippets::op::BroadcastMove>(node) && !ngraph::is_type<ngraph::snippets::op::Load>(node)) {
        return;
    }

    fix_concatenation(node, register_number);

    fix_concatenation_up(node->get_input_node_shared_ptr(0), register_number);
}

void fix_concatenation_down(const std::shared_ptr<ngraph::Node>& node, const size_t register_number) {
    for (auto output : node->outputs()) {
        const auto& target_inputs = output.get_target_inputs();
        for (auto target_input : target_inputs) {
            auto target_node = target_input.get_node()->shared_from_this();
            if (!ngraph::is_type<ngraph::snippets::op::BroadcastMove>(target_node) && !ngraph::is_type<ngraph::snippets::op::Load>(target_node)) {
                return;
            }

            fix_concatenation(target_node, register_number);

            fix_concatenation_down(target_node, register_number);
        }
    }
}
} // namespace

bool ngraph::snippets::pass::AssignRegisters::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(AssignRegisters);
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::AssignRegisters")
    using Reg = size_t;
    auto ops = f->get_ordered_ops();
    decltype(ops) stmts;
    std::copy_if(ops.begin(), ops.end(), std::back_inserter(stmts), [](decltype(ops[0]) op) {
        return !(std::dynamic_pointer_cast<opset1::Parameter>(op) || std::dynamic_pointer_cast<opset1::Result>(op));
        });

    size_t rdx = 0;
    std::map<std::shared_ptr<descriptor::Tensor>, Reg> regs;
    for (const auto& op : stmts) {
        for (const auto& output : op->outputs()) {
            regs[output.get_tensor_ptr()] = rdx++;
        }
    }

    std::vector<std::set<Reg>> used;
    std::vector<std::set<Reg>> def;

    for (const auto& op : stmts) {
        std::set<Reg> u;
        for (const auto& input : op->inputs()) {
            if (regs.count(input.get_tensor_ptr())) {
                u.insert(regs[input.get_tensor_ptr()]);
            }
        }
        used.push_back(u);

        std::set<Reg> d;
        if (!std::dynamic_pointer_cast<snippets::op::Store>(op)) {
            for (const auto& output : op->outputs()) {
                d.insert(regs[output.get_tensor_ptr()]);
            }
        }
        def.push_back(d);
    }

    // define life intervals
    std::vector<std::set<Reg>> lifeIn(stmts.size(), std::set<Reg>());
    std::vector<std::set<Reg>> lifeOut(stmts.size(), std::set<Reg>());

    for (size_t i = 0; i < stmts.size(); i++) {
        for (size_t n = 0; n < stmts.size(); n++) {
            std::set_difference(lifeOut[n].begin(), lifeOut[n].end(), def[n].begin(), def[n].end(), std::inserter(lifeIn[n], lifeIn[n].begin()));
            lifeIn[n].insert(used[n].begin(), used[n].end());
        }
        for (size_t n = 0; n < stmts.size(); n++) {
            auto node = stmts[n];
            if (!std::dynamic_pointer_cast<snippets::op::Store>(node)) {
                for (const auto& out : node->outputs()) {
                    for (const auto& port : out.get_target_inputs()) {
                        auto pos = std::find(stmts.begin(), stmts.end(), port.get_node()->shared_from_this());
                        if (pos != stmts.end()) {
                            auto k = pos-stmts.begin();
                            lifeOut[n].insert(lifeIn[k].begin(), lifeIn[k].end());
                        }
                    }
                }
            }
        }
    }

    struct by_starting {
        auto operator()(const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) const -> bool {
            return lhs.first < rhs.first|| (lhs.first == rhs.first && lhs.second < rhs.second);
        }
    };

    struct by_ending {
        auto operator()(const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) const -> bool {
            return lhs.second < rhs.second || (lhs.second == rhs.second && lhs.first < rhs.first);
        }
    };

    std::set<std::pair<int, int>, by_starting> live_intervals;

    std::reverse(lifeIn.begin(), lifeIn.end());
    auto find_last_use = [lifeIn](int i) -> int {
        int ln = lifeIn.size()-1;
        for (auto& x : lifeIn) {
            if (x.find(i) != x.end()) {
                return ln;
            }
            ln--;
        }
        return i;
    };

    for (size_t i = 0; i < stmts.size(); i++) {
        live_intervals.insert(std::make_pair(i, find_last_use(i)));
    }

    // http://web.cs.ucla.edu/~palsberg/course/cs132/linearscan.pdf
    std::multiset<std::pair<int, int>, by_ending> active;
    std::map<Reg, Reg> register_map;
    std::stack<Reg> bank;
    for (int i = 0; i < 16; i++) bank.push(16-1-i);

    for (auto interval : live_intervals) {
        // check expired
        while (!active.empty()) {
            auto x = *active.begin();
            if (x.second >= interval.first) {
                break;
            }
            active.erase(x);
            bank.push(register_map[x.first]);
        }
        // allocate
        if (active.size() == 16) {
            throw ngraph_error("caanot allocate registers for a snippet ");
        } else {
            register_map[interval.first] = bank.top();
            bank.pop();
            active.insert(interval);
        }
    }

    std::map<std::shared_ptr<descriptor::Tensor>, Reg> physical_regs;

    for (const auto& reg : regs) {
        physical_regs[reg.first] = register_map[reg.second];
    }
    const auto num_parameters = f->get_parameters().size();

    auto as_parameter = [](const std::shared_ptr<Node>& source) -> std::shared_ptr<opset1::Parameter> {
        auto parameter = ov::as_type_ptr<opset1::Parameter>(source);
        if (parameter != nullptr) {
            return parameter;
        }

        const auto split = ov::as_type_ptr<opset1::Split>(source);
        if (split == nullptr) {
            return nullptr;
        }

        // TODO: should check if parameter is assigned with Constant
        parameter = ov::as_type_ptr<opset1::Parameter>(split->get_input_node_shared_ptr(0));
        if (parameter == nullptr) {
            return nullptr;
        }

        return parameter;
    };

    for (const auto& n : f->get_ordered_ops()) {
        auto& rt = n->get_rt_info();
        std::vector<size_t> regs;
        regs.reserve(n->outputs().size());
        /* The main idea here is that each operation stores its output regs in rt["reginfo"]. Input and output regs are
         * then derived by parsing node's and parent's rt["reginfo"], look into ngraph::snippets::getRegisters for details.
         * Note also that Parameter and Result store general-purpose register index, because they work with memory
         * (memory pointer is stored in gpr). All other "regular" ops store vector regs indexes, since calculations are
         * performed on registers.
         */
        if (is_type<ov::op::v0::Result>(n)) {
            continue;
        } else if (const auto& param = as_parameter(n)) {
            regs.push_back(f->get_parameter_index(param));
        } else if (const auto& store = ov::as_type_ptr<ngraph::snippets::op::Store>(n)) {
            regs.push_back(f->get_result_index(store) + num_parameters);
        } else {
            for (const auto& output : n->outputs()) {
                auto allocated = physical_regs[output.get_tensor_ptr()];
                regs.push_back(allocated);
            }
        }

        rt["reginfo"] = regs;
    }

    // TODO: snippets fix
    //for (const auto& n : f->get_ordered_ops()) {
    //    auto& rt = n->get_rt_info();
    //    auto it = rt.find("reginfo");
    //    if (it == rt.end()) {
    //        continue;
    //    }
    //
    //    auto regs = it->second.as<std::vector<size_t>>();
    //    const auto& outputs = n->outputs();
    //    for (auto i = 0; i < outputs.size(); ++i) {
    //        const auto& target_inputs = outputs[i].get_target_inputs();
    //        if (target_inputs.size() != 1ul) {
    //            // TODD: snippets: not supported
    //            continue;
    //        }
    //
    //        auto target = target_inputs.begin()->get_node();
    //    }
    //}

    //auto update_concatetion_by_pattern = [](const std::shared_ptr<ov::Model>& f) {
    //    auto is_concatenation_split = [](const std::shared_ptr<Node>& n) {
    //        if (is_type<opset1::Split>(n) &&
    //            is_type<opset1::Parameter>(n->get_input_node_shared_ptr(0)) &&
    //            is_type<opset1::Constant>(n->get_input_node_shared_ptr(1))) {
    //            auto get_child = [](const ngraph::Output<Node>& output) -> ngraph::Node* {
    //                const auto& target_inputs = output.get_target_inputs();
    //                if (target_inputs.size() > 1ul) {
    //                    return nullptr;
    //                }
    //                return target_inputs.begin()->get_node();
    //            };
    //
    //            for (const auto output : n->outputs()) {
    //                const auto& load = as_type<snippets::op::Load>(get_child(output));
    //                if (load == nullptr) {
    //                    return false;
    //                }
    //
    //                const auto& broadcast_move = as_type<snippets::op::BroadcastMove>(get_child(load->output(0)));
    //                if (broadcast_move == nullptr) {
    //                    return false;
    //                }
    //            }
    //            return true;
    //        }
    //
    //        return false;
    //    };
    //
    //
    //    size_t max_register = -1;
    //    for (const auto& n : f->get_ordered_ops()) {
    //        if (is_concatenation_split(n)) {
    //            if (max_register == -1) {
    //                max_register = get_max_register(f);
    //            }
    //            max_register++;
    //            fix_concatenation_down(n, max_register);
    //        }
    //    }
    //
    //    size_t max_constant_register = -1;
    //    for (const auto& n : f->get_ordered_ops()) {
    //        const auto input_size = n->get_input_size();
    //        if (input_size == 1ul) {
    //            continue;
    //        }
    //
    //        std::unordered_map<size_t, std::shared_ptr<Node>> register_by_parent_node;
    //        for (auto i = 0; i < input_size; ++i) {
    //            const auto& source_output = n->input(i).get_source_output();
    //            const auto index = source_output.get_index();
    //            const auto& source = source_output.get_node_shared_ptr();
    //
    //            auto& rt = source->get_rt_info();
    //            auto it = rt.find("reginfo");
    //            if (it == rt.end()) {
    //                continue;
    //            }
    //
    //            auto source_registers = it->second.as<std::vector<size_t>>();
    //            auto source_register = source_registers[index];
    //
    //            auto existing_it = register_by_parent_node.find(source_register);
    //            if (existing_it != register_by_parent_node.end()) {
    //                if (max_constant_register == -1) {
    //                    // 7ul
    //                    max_constant_register = get_max_constant_register(f);
    //                }
    //                max_constant_register++;
    //                fix_concatenation(source, max_constant_register);
    //            }
    //
    //            register_by_parent_node.emplace(source_register, source);
    //        }
    //    }
    //};
    //
    //update_concatetion_by_pattern(f);


    auto update_concatenation_by_branch = [](const std::shared_ptr<ov::Model>& f) {
        auto is_concatenation_load_branch = [](const std::shared_ptr<Node> &node) -> bool {
            if (!ngraph::is_type<snippets::op::BroadcastMove>(node)) {
                return false;
            }

            const auto load = node->get_input_source_output(0).get_node_shared_ptr();
            if (!ngraph::is_type<snippets::op::Load>(load)) {
                return false;
            }

            const auto split = load->get_input_source_output(0).get_node_shared_ptr();
            if (!ngraph::is_type<opset1::Split>(split)) {
                return false;
            }

            return true;
        };

        for (const auto &n : f->get_ordered_ops()) {
            const auto input_size = n->get_input_size();
            if (input_size == 1ul) {
                continue;
            }

            std::unordered_map<size_t, std::shared_ptr<Node>> register_by_parent_node;
            std::shared_ptr<Node> prev_source = nullptr;

            for (auto i = 0; i < input_size; ++i) {
                const auto &source_output = n->input(i).get_source_output();
                const auto &source = source_output.get_node_shared_ptr();
                if (ngraph::is_type<opset1::Parameter>(source)) {
                    continue;
                }

                auto &rt = source->get_rt_info();
                auto it = rt.find("reginfo");
                if (it == rt.end()) {
                    continue;
                }

                const auto& source_registers = it->second.as<std::vector<size_t>>();
                auto source_register = source_registers[source_output.get_index()];

                if (prev_source != nullptr) {
                    auto get_free_register = [](std::unordered_map<size_t, std::shared_ptr<Node>>& register_by_parent_node) -> size_t {
                        // TODO: snippets: define more appropriate value
                        const size_t max_register = 7;
                        for (size_t i = 0ul; i <= max_register; ++i) {
                            if (register_by_parent_node.find(i) == register_by_parent_node.end()) {
                                return i;
                            }
                        }
                        throw ov::Exception("all registers are occupied");
                    };

                    auto fix_concatenation_load_branch = [&](
                            std::shared_ptr<Node> source,
                            std::unordered_map<size_t, std::shared_ptr<Node>>& register_by_parent_node) -> size_t {
                        const auto free_register = get_free_register(register_by_parent_node);
                        fix_concatenation_up(source, free_register);
                        return free_register;
                    };

                    auto fix_branch = [&](
                            std::shared_ptr<Node> source,
                            std::unordered_map<size_t, std::shared_ptr<Node>>& register_by_parent_node) -> size_t {
                        const auto free_register = get_free_register(register_by_parent_node);
                        fix_concatenation(source, free_register);
                        return free_register;
                    };

                    if ((register_by_parent_node.size() == 1ul) && is_concatenation_load_branch(prev_source)) {
                        // the first parent node was skipped before
                        const auto prev_source_register = fix_concatenation_load_branch(prev_source, register_by_parent_node);
                        register_by_parent_node.emplace(prev_source_register, prev_source);
                    }

                    if (is_concatenation_load_branch(source)) {
                        source_register = fix_concatenation_load_branch(source, register_by_parent_node);
                    } else {
                        if (register_by_parent_node.find(source_register) != register_by_parent_node.end()) {
                            source_register = fix_branch(source, register_by_parent_node);
                        }
                    }
                }

                register_by_parent_node.emplace(source_register, source);
                prev_source = source;
            }
        }
    };

    update_concatenation_by_branch(f);

    {
        auto index = 1ul;
        for (auto node : f->get_ordered_ops()) {
            auto &rt_info = node->get_rt_info();
            rt_info["order"] = index;
            ++index;
        }
    }

    ov::pass::VisualizeTree("svg/snippets.assign_registers.svg").run_on_model(f);

    return false;
}
