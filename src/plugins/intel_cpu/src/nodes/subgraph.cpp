// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph.h"

#include <ie_parallel.hpp>

#include <vector>
#include <algorithm>
#include <array>
#include <tuple>

#include <dnnl_debug.h>
#include <onednn/dnnl.h>
#include <dnnl_extension_utils.h>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/visualize_tree.hpp>
#include <ngraph/rt_info.hpp>
#include <ie_ngraph_utils.hpp>

#include <memory_desc/dnnl_blocked_memory_desc.h>
#include <snippets/op/subgraph.hpp>
#include <snippets/roi_backprop/roi_backprop.hpp>

#include "emitters/cpu_generator.hpp"

using namespace InferenceEngine;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

namespace ov {
namespace intel_cpu {
namespace node {

Snippet::Snippet(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache)
        : Node(op, eng, cache) {
    host_isa = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_common) ?
        dnnl::impl::cpu::x64::avx512_common : dnnl::impl::cpu::x64::avx2;

    // Create a deep local copy of the input snippet to perform canonicalization & code generation
    // Todo: Probably better to implement a proper copy constructor
    if (const auto tmp_snippet =  ov::as_type_ptr<ngraph::snippets::op::Subgraph>(op)) {
        ngraph::OutputVector subgraph_node_inputs;
        for (const auto &input : tmp_snippet->input_values()) {
            auto new_input = std::make_shared<ngraph::opset1::Parameter>(input.get_element_type(), input.get_partial_shape());
            subgraph_node_inputs.push_back(new_input);
        }
        auto new_body = ov::clone_model(*tmp_snippet->get_body().get());
        snippet = std::make_shared<ngraph::snippets::op::Subgraph>(subgraph_node_inputs, new_body);
        ngraph::copy_runtime_info(tmp_snippet, snippet);
        snippet->set_friendly_name(tmp_snippet->get_friendly_name());
        snippet->set_generator(std::make_shared<CPUGenerator>(host_isa));
    } else {
        IE_THROW(NotImplemented) << "Node is not an instance of snippets::op::Subgraph";
    }
}

void Snippet::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    const Precision supportedPrecision = Precision::FP32;

    bool dimRanksAreEqual = true;
    // TODO: handle weights
    //for (size_t i = 0; dimRanksAreEqual && i < inputShapes.size(); i++) {
    //    for (size_t j = 0; dimRanksAreEqual && j < outputShapes.size(); j++) {
    //        if (inputShapes[i].getRank() != outputShapes[j].getRank())
    //            dimRanksAreEqual = false;
    //    }
    //}

    const size_t ndims = outputShapes[0].getRank();
    const bool isChannelsFirstApplicable = dnnl::impl::utils::one_of(ndims, 1, 2, 4, 5) && dimRanksAreEqual;
    // Todo: Snippets currently don't support per-channel broadcasting of Blocked descriptors because
    //  canonicalization can't distinguish between <N, C, H, W, c> and <N, C, D, H, W> cases.
    //  See snippets::op::Subgraph::canonicalize for details.
    const bool isBlockedApplicable = dnnl::impl::utils::one_of(ndims,  4, 5) && dimRanksAreEqual;
    enum LayoutType {
        Planar,
        ChannelsFirst,
        Blocked
    };
    auto initDesc = [&] (LayoutType lt) -> NodeDesc {
        auto createMemoryDesc = [lt](const Shape &shape, Precision prc, size_t offset) -> std::shared_ptr<CpuBlockedMemoryDesc> {
            const auto &dims = shape.getDims();
            if (lt == ChannelsFirst && shape.getRank() != 1) {
                auto ndims = shape.getRank();
                VectorDims order(ndims);
                std::iota(order.begin(), order.end(), 0);
                if (ndims > 1) {
                    order.erase(order.begin() + 1);
                    order.push_back(1);
                }

                VectorDims blocks(ndims);
                for (size_t i = 0; i < order.size(); i++) {
                    blocks[i] = dims[order[i]];
                }

                return std::make_shared<CpuBlockedMemoryDesc>(prc, shape, blocks, order, offset);
            } else if (lt == Blocked && shape.getRank() != 1 && (shape.getMinDims()[1] != Shape::UNDEFINED_DIM && shape.getMinDims()[1] > 1)) {
                size_t blockSize = mayiuse(dnnl::impl::cpu::x64::avx512_common) ? 16 : 8;

                VectorDims blocks = dims;
                VectorDims order(blocks.size());
                std::iota(order.begin(), order.end(), 0);

                blocks[1] = dims[1] != Shape::UNDEFINED_DIM ? div_up(blocks[1], blockSize) : Shape::UNDEFINED_DIM;
                blocks.push_back(blockSize);
                order.push_back(1);

                return std::make_shared<CpuBlockedMemoryDesc>(prc, shape, blocks, order, offset);
            } else {
                VectorDims blocks = dims;
                VectorDims order(blocks.size());
                std::iota(order.begin(), order.end(), 0);

                return std::make_shared<CpuBlockedMemoryDesc>(prc, shape, blocks, order, offset);
            }
        };

        size_t offset = 0;
        NodeConfig config;
        config.dynBatchSupport = false;
        config.inConfs.resize(inputShapes.size());
        for (size_t i = 0; i < inputShapes.size(); i++) {
            BlockedMemoryDesc::CmpMask inputMask = BLOCKED_DESC_SKIP_OFFSET_MASK;
            PortConfig portConfig;
            portConfig.inPlace((!i && canBeInPlace()) ? 0 : -1);
            portConfig.constant(false);
            if (inputShapes[i].getDims()[0] == 1) {
                inputMask.reset(0); // accepts any stride on batch axis
            }

            if (i == 1ul) {
                auto shape = getInputShapeAtPort(1);
                auto mem_desc = std::make_shared<ov::intel_cpu::DnnlBlockedMemoryDesc>(
                        shape,
                        dnnl::memory::data_type::f32,
                        dnnl::memory::format_tag::OIhw8i8o);

                std::shared_ptr<BlockedMemoryDesc> blocked_mem_desc = std::dynamic_pointer_cast<BlockedMemoryDesc>(mem_desc);
                portConfig.setMemDesc(blocked_mem_desc);
            } else if (i == 3ul) {
                //auto shape = getInputShapeAtPort(1);
                auto shape = inputShapes[i];
                auto mem_desc = std::make_shared<ov::intel_cpu::DnnlBlockedMemoryDesc>(
                    shape,
                    dnnl::memory::data_type::f32,
                    dnnl::memory::format_tag::OIdhw8i8o);

                std::shared_ptr<BlockedMemoryDesc> blocked_mem_desc = std::dynamic_pointer_cast<BlockedMemoryDesc>(mem_desc);
                portConfig.setMemDesc(blocked_mem_desc);
            } else {
                portConfig.setMemDesc(createMemoryDesc(inputShapes[i], supportedPrecision, offset), inputMask);
            }
            config.inConfs[i] = portConfig;
        }
        config.outConfs.resize(outputShapes.size());
        for (size_t i = 0; i < outputShapes.size(); i++) {
            BlockedMemoryDesc::CmpMask outputMask = BLOCKED_DESC_SKIP_OFFSET_MASK;
            PortConfig portConfig;
            portConfig.inPlace(-1);
            portConfig.constant(false);
            if (outputShapes[i].getDims()[0] == 1) {
                outputMask.reset(0); // accepts any stride on batch axis
            }
            portConfig.setMemDesc(createMemoryDesc(outputShapes[i], supportedPrecision, offset), outputMask);
            config.outConfs[i] = portConfig;
        }

        impl_desc_type impl_type = impl_desc_type::unknown;
        if (mayiuse(x64::avx512_common)) {
            impl_type = impl_desc_type::jit_avx512;
        } else if (mayiuse(x64::avx2)) {
            impl_type = impl_desc_type::jit_avx2;
        }
        return {config, impl_type};
    };

    if (isChannelsFirstApplicable)
        supportedPrimitiveDescriptors.emplace_back(initDesc(ChannelsFirst));
    if (isBlockedApplicable)
        supportedPrimitiveDescriptors.emplace_back(initDesc(Blocked));
    supportedPrimitiveDescriptors.emplace_back(initDesc(Planar));
}

void Snippet::selectOptimalPrimitiveDescriptor() {
    selectPreferPrimitiveDescriptor(getPrimitivesPriority(), true);
}

void Snippet::createPrimitive() {
    // schedule definition part
    // it defines offsets, strides and sizes for snippet kernel scheduling
    define_schedule();

    // code generation part
    // it might be worth to generate explicitly for scheduler work amount for now,
    // but in future some interface should be defined in order to communicate schedule for a kernel
    // or generate schedule for a kernel.
    // Here kernel is generated for most warying dimension by default.
    generate();
}

void Snippet::execute(dnnl::stream strm) {
    if (schedule.ptr == nullptr || !canUseOptimizedImpl) {
        IE_THROW() << "Snippet can't use Optimized implementation and can't fallback to reference";
    }
    jit_snippets_call_args call_args;
    for (size_t i = 0; i < srcMemPtrs.size(); i++)
        call_args.src_ptrs[i] = reinterpret_cast<const uint8_t*>(srcMemPtrs[i]->GetData()) + start_offset_in[i];

    for (size_t i = 0; i < dstMemPtrs.size(); i++)
        call_args.dst_ptrs[i] = reinterpret_cast<uint8_t*>(dstMemPtrs[i]->GetData()) + start_offset_out[i];

#ifdef CPU_DEBUG_CAPS
    // TODO: backprop: debug only
    auto display = [](std::vector<ov::intel_cpu::MemoryPtr>& memPtrs) {
        for (size_t i = 0; i < memPtrs.size(); i++) {
            float* value = reinterpret_cast<float*>(memPtrs[i]->GetData());
            auto shape = memPtrs[i]->GetShape().getDims();
            std::cout << std::endl << "memPtrs[i]: i=" << i << ", shape=" << shape << std::endl;

            const auto spacial_volume = shape[2] * shape[3];
            for (auto c = 0ull; c < shape[1]; ++c) {
                std::cout << std::endl << "channel: " << c;
                auto h = 0ul;
                for (auto w = 0ull; w < spacial_volume; ++w) {
                    if ((w % shape[2]) == 0ul) {
                        std::cout << std::endl << h << ": ";
                        h++;
                    }

                    std::cout << "\t" << value[w * 8 + (c % 8) + (c / 8) * spacial_volume * 8];
                }
            }
        }
        std::cout << std::endl;
    };

    //// TODO: backprop: debug only
    //auto display_raw = [](std::vector<ov::intel_cpu::MemoryPtr>& memPtrs) {
    //    for (size_t i = 0; i < memPtrs.size(); i++) {
    //        float* value = reinterpret_cast<float*>(memPtrs[i]->GetData());
    //        auto shape = memPtrs[i]->GetShape().getDims();
    //        std::cout << std::endl << "memPtrs[i]: i=" << i << ", shape=" << shape << std::endl;
    //
    //        const auto spacial_volume = shape[2] * shape[3];
    //        for (auto i = 0; i < spacial_volume * shape[1]; ++i) {
    //            std::cout << "\t" << i << ": " << value[i];
    //        }
    //    }
    //    std::cout << std::endl;
    //};
#endif

#ifdef CPU_DEBUG_CAPS
    std::cout << std::endl << "srcMemPtrs.size() = " << srcMemPtrs.size() << std::endl;
    display(srcMemPtrs);
#endif

    if (tensorRank == rank6D) {
        schedule_6d(call_args);
    } else {
        schedule_nt(call_args);
    }

#ifdef CPU_DEBUG_CAPS
    std::cout << std::endl << "dstMemPtrs.size() = " << dstMemPtrs.size() << std::endl;
    display(dstMemPtrs);
#endif
}

bool Snippet::created() const {
    return getType() == Type::Subgraph;
}

bool Snippet::canBeInPlace() const {
    if (getParentEdgesAtPort(0)[0]->getParent()->getType() == Type::Input) {
        return false;
    }

    for (auto& parentEdge : getParentEdges()) {
        auto parent = parentEdge.lock()->getParent();
        if (parent->getChildEdges().size() != 1)
            return false;

        // WA to prevent memory corruption caused by inplace feature
        if (parent->getType() == Type::Concatenation) {
            for (auto& parentParentEdge : parent->getParentEdges()) {
                auto parentParent = parentParentEdge.lock()->getParent();
                if (parentParent->getChildEdges().size() != 1)
                    return false;
            }
        }
    }
    return getInputShapeAtPort(0) == getOutputShapeAtPort(0);
}

static void offset_calculation(std::vector<size_t>& offset, const std::vector<size_t>& dims_in, const std::vector<size_t>& dims_out) {
    // TODO: backprop: question: looks like offsets calculated not correctly for different in/out (exec_domain) dimensions
    // in:  {1, 1, 1, 64, 64, 8}
    // out: {1, 1, 1, 32, 32, 8}
    // offsets: {32768, 32768, 0, 0, 1}
    // expected: offset doesn't depend on output dims
    // expected: {32768, 32768, 32768, 512, 8, 1}

    // stride calculation
    // <= for layout oblivious operation only


    // two types offset
    size_t k = 1;
    for (int i = offset.size() - 1; i >= 0; i--) {
        offset[i] = (dims_in[i] == dims_out[i]) ? k : 0;
        k *= dims_in[i];
    }
}

static void offset_calculation(std::vector<size_t>& offset, const std::vector<size_t>& dims_in) {
    size_t k = 1;
    for (int i = offset.size() - 1; i >= 0; i--) {
        offset[i] = k;
        k *= dims_in[i];
    }
}

static auto collapseLastDims(std::vector<size_t>& dims, size_t dimsToCollapse) -> void {
    if (dimsToCollapse >= dims.size() - 1)
        IE_THROW() << "Got invalid number of dims to collapse. Expected < " << dims.size() - 1 << " got " << dimsToCollapse;
    for (int i = dims.size() - 2; i > dims.size() - dimsToCollapse - 2; i--) {
        dims[dims.size() - 1] *= dims[i];
    }

    for (int i = dims.size() - 2; i >= dimsToCollapse; i--) {
        dims[i] = dims[i - dimsToCollapse];
    }

    for (int i = dimsToCollapse - 1; i >= 0; i--) {
        dims[i] = 1;
    }
}

void Snippet::define_schedule() {
    auto edgeToBlockedShape = [](const EdgePtr& edge) {
        const auto blockedDesc = edge->getMemory().GetDescWithType<BlockedMemoryDesc>();
        ngraph::Shape shape(blockedDesc->getBlockDims());
        ngraph::AxisVector blocking(blockedDesc->getOrder());
        ngraph::element::Type precision = InferenceEngine::details::convertPrecision(blockedDesc->getPrecision());
        return ngraph::snippets::op::Subgraph::BlockedShape{shape, blocking, precision};
    };
    auto prependWithOnes = [this](const std::vector<size_t>& dims) {
        if (tensorRank <= dims.size())
            return dims;
        VectorDims result(tensorRank, 1);
        std::copy(dims.begin(), dims.end(), &result[tensorRank - dims.size()]);
        return result;
    };
    ngraph::snippets::op::Subgraph::BlockedShapeVector input_blocked_shapes;
    for (size_t i = 0; i < inputShapes.size(); i++) {
        const auto& parentEdgesAtPort = getParentEdgesAtPort(i);
        const auto& parentEdgeAtPort = parentEdgesAtPort[0];
        //const auto& child = parentEdgeAtPort->getChild();
        //const auto type = child->getType();
        //if (type == ov::intel_cpu::Type::Convolution) {
        //    continue;
        //}

        const auto blockedDesc = parentEdgeAtPort->getMemory().GetDescWithType<BlockedMemoryDesc>();
        auto dims = blockedDesc->getBlockDims();

        auto result = edgeToBlockedShape(parentEdgeAtPort);
        input_blocked_shapes.push_back(result);

        //input_blocked_shapes.push_back(edgeToBlockedShape(getParentEdgesAtPort(i)[0]));
    }

    ngraph::snippets::op::Subgraph::BlockedShapeVector output_blocked_shapes;
    for (size_t i = 0; i < outputShapes.size(); i++)
        output_blocked_shapes.push_back(edgeToBlockedShape(getChildEdgesAtPort(i)[0]));
    exec_domain = snippet->canonicalize(output_blocked_shapes, input_blocked_shapes);
    // initialize by maximum output dimension. Dimensions of outputs should be broadcastable
    tensorRank = std::max(static_cast<size_t>(rank6D), exec_domain.size());
    // Canonicalization broadcasts inputs and outputs to max input rank, which can be smaller than tensorRank
    // prepend to enable 6D scheduler
    exec_domain = prependWithOnes(exec_domain);
    const auto &body = snippet->get_body();
    for (const auto& p : body->get_parameters()) {
        dims_in.emplace_back(prependWithOnes(p->get_shape()));
    }

    for (size_t i = 0; i < body->get_output_size(); i++) {
        dims_out.push_back(prependWithOnes(body->get_output_shape(i)));
    }

    const auto config = getSelectedPrimitiveDescriptor()->getConfig();
    const auto dataSize = config.inConfs[0].getMemDesc()->getPrecision().size();
    auto initOffsets = [this, config, dataSize]() {
        auto body = this->snippet->get_body();
        // TODO: bakprop: hardcode
        std::vector<size_t> unit_shape;
        const auto& results = body->get_results();
        assert(results.size() == 1ul);
        unit_shape.resize(results[0]->get_shape().size(), 1ul);
        // TODO: backprop: add strides
        std::map<ov::Node*, ov::snippets::ROIBackprop> map = ov::snippets::get_roi_from_function(body, {PartialShape(unit_shape)});

        const auto& params = body->get_parameters();
        std::vector<ov::snippets::ROIBackprop> param_roi;
        param_roi.reserve(params.size());
        for (const auto& param : params) {
            //std::shared_ptr<ov::Node> node = std::dynamic_pointer_cast<ov::Node>(param);
            ov::Node* node2 = param.get();
            auto it = map.find(node2);
            if (it == map.end()) {
                // TODO: throw exception
            }

            auto roi_shape = it->second;
            param_roi.push_back(roi_shape);
        }

        // find max rank input among all outputs
        const size_t inputNum = getParentEdges().size();
        offsets_in.resize(inputNum);

        //dims_in[0] = {1, 1, 1, 32, 32, 8};

        // {1, 1, 1, 64, 64, 8}
        for (size_t i = 0; i < inputNum; i++) {
            offsets_in[i].resize(tensorRank, 1);

            // TODO: backprop: fix here for input & output
            // input:
            //    offsets_in: {1, 1, 1, 1, 1, 1}
            //    dims_in:  {1, 1, 1, 32, 32, 8}
            //    exec_domain (dims_out): {1, 1, 1, 16, 16, 8}
            // result:
            //    offsets_in: {8192, 8192, 8192, 0, 0, 1}


            // input:
            //    offsets_in: {1, 1, 1, 1, 1, 1}
            //    dims_in:  {1, 1, 1, 16, 16, 8}
            //    exec_domain (dims_out): {1, 1, 1, 16, 16, 8}
            // result:
            //    offsets_in: {2048, 2048, 2048, 128, 8, 1}

            // TODO: just to test
            //const std::vector<std::size_t> roi_shape_shift = { 4, 2 };
            const auto& roi = param_roi[i];
            //const auto& roi_shape = roi.shapes[0].get_shape();
            const auto& roi_strides = roi.strides[0];

            if (i == 0) {
                offset_calculation(offsets_in[i], dims_in[i]);

                offsets_in[0] =  {
                        offsets_in[0][0] * 1ul, // TODO: backprop: question: what does this dimension mean?
                        offsets_in[0][1] * roi_strides[0ul],
                        offsets_in[0][2] * roi_strides[1ul],
                        offsets_in[0][3] * roi_strides[2ul],
                        offsets_in[0][4] * roi_strides[3ul],
                        offsets_in[0][5] * roi_strides[4ul]};

            } else {
                offset_calculation(offsets_in[i], dims_in[i], exec_domain);
            }

            for (size_t j = 0; j < tensorRank; j++) {
                offsets_in[i][j] *= dataSize;
            }
        }

        // TODO: backprop: question: soffsets hardcoded
        //offsets_in[0] = { 1, 8, 256 * 4, 8192 * 4, 8192 * 4, 8192 * 4};

        start_offset_in.resize(inputNum);
        srcMemPtrs.resize(inputNum);
        for (size_t i = 0; i < inputNum; i++) {
            const auto memPtr = getParentEdgeAt(i)->getMemoryPtr();
            srcMemPtrs[i] = memPtr;
            start_offset_in[i] =  memPtr->GetDescWithType<BlockedMemoryDesc>()->getOffsetPadding() * dataSize;
        }

        const size_t outputNum = config.outConfs.size();
        offsets_out.resize(outputNum);
        for (size_t i = 0; i < outputNum; i++) {
            offsets_out[i].resize(tensorRank, 1);
            offset_calculation(offsets_out[i], dims_out[i], exec_domain);
            for (size_t j = 0; j < tensorRank; j++) {
                offsets_out[i][j] *= dataSize;
            }
        }

        start_offset_out.resize(outputNum);
        dstMemPtrs.resize(outputNum);
        for (size_t i = 0; i < outputNum; i++) {
            const auto memPtr = getChildEdgeAt(i)->getMemoryPtr();
            dstMemPtrs[i] = memPtr;
            start_offset_out[i] = memPtr->GetDescWithType<BlockedMemoryDesc>()->getOffsetPadding() * dataSize;
        }
    };

    auto find_dims_to_collapse = [this, config]() -> int {
        int collapsedDims = 0;
        size_t minimalConcurrency = parallel_get_max_threads();
        size_t minimalJitWorkAmount = 256;
        size_t currentJitWorkAmount = exec_domain.back();
        while (currentJitWorkAmount < minimalJitWorkAmount && currentJitWorkAmount < fullWorkAmount) {
            if (static_cast<int>(exec_domain.size()) - collapsedDims - 2 < 0)
                break;

            bool canCollapse = true;
            for (size_t i = 0; i < dims_in.size(); i++) {
                if ((dims_in[i][dims_in[i].size() - 2] != 1 && dims_in[i][dims_in[i].size() - 1] == 1) ||
                    (dims_in[i][dims_in[i].size() - 2] == 1 && dims_in[i][dims_in[i].size() - 1] != 1)) {
                    canCollapse = false;
                    break;
                }
            }

            size_t nextJitWorkAmount = currentJitWorkAmount * exec_domain[exec_domain.size() - 2];
            if (fullWorkAmount / nextJitWorkAmount >= minimalConcurrency) {
                currentJitWorkAmount = nextJitWorkAmount;
                // if we cannot use dim collapsing we should use tile2D
                if (!canCollapse) {
                    if (tileRank < maxTileRank) {
                        tileRank++;
                        continue;
                    }

                    break;
                }

                collapsedDims++;
                for (auto &d : dims_in)
                    collapseLastDims(d, 1);

                for (auto &d : dims_out)
                    collapseLastDims(d, 1);

                collapseLastDims(exec_domain, 1);
            } else {
                break;
            }
        }
        return collapsedDims;
    };

    auto initSchedulingInfo = [this, dataSize]() -> void {
        // initialize scheduling information
        sch_offsets_in.resize(offsets_in.size(), 0);
        sch_offsets_out.resize(offsets_out.size(), 0);
        sch_dims.resize(maxTileRank, 1);
        sch_dims[maxTileRank-1] = exec_domain.back();
        schedulerWorkAmount = fullWorkAmount / exec_domain.back();
        if (tileRank > 1) {
            // TODO: backprop: scheduler dimensions 0 index (outer_work_amount) can be not correct
            sch_dims[maxTileRank - tileRank] = exec_domain[tensorRank - 2];
            schedulerWorkAmount /= exec_domain[tensorRank - 2];
            exec_domain[tensorRank - 2] = 1;

            // update offsets for tile 2D because loaders have ptr shifts in some cases and stores have always ptrs shifts
            for (size_t i = 0; i < offsets_in.size(); i++) {
                int64_t offset = offsets_in[i][tensorRank - 2];
                if ((offset > dataSize) || (offset == 0 && dims_in[i].back() != 1)) {
                    sch_offsets_in[i] = offset - exec_domain.back() * dataSize;
                } else if (offset == dataSize) {
                    sch_offsets_in[i] = offset;
                }
            }

            for (size_t i = 0; i < offsets_out.size(); i++) {
                int64_t offset = offsets_out[i][tensorRank - 2];
                sch_offsets_out[i] = offset - exec_domain.back() * dataSize;
            }
        }
    };

    fullWorkAmount = 1;
    for (const auto &d : exec_domain) {
        fullWorkAmount *= d;
    }

    // TODO: backprop: question: hardcoded value
    //fullWorkAmount = fullWorkAmount / 2;

    batchDimIdx = tensorRank - exec_domain.size();
    // Note that exec_domain can be modified inside find_dims_to_collapse() and/or initSchedulingInfo()
    find_dims_to_collapse();

    initOffsets();
    initSchedulingInfo();
}

void Snippet::generate() {
    jit_snippets_compile_args jcp;
    jcp.output_dims = exec_domain;
    std::copy(sch_dims.begin(), sch_dims.end(), jcp.scheduler_dims);
    std::copy(sch_offsets_in.begin(), sch_offsets_in.end(), jcp.scheduler_offsets);

#ifdef CPU_DEBUG_CAPS
    // TODO: backprop: debug only
    std::cout << "jcp.scheduler_offsets:" << std::endl;
    for (auto i = 0ul; i < SNIPPETS_MAX_SNIPPETS_DIMS; ++i) {
        std::cout << "\ti: " << jcp.scheduler_offsets[i] << std::endl;
    }
#endif

    std::copy(sch_offsets_out.begin(), sch_offsets_out.end(), &jcp.scheduler_offsets[sch_offsets_in.size()]);
    size_t harness_num_dims = jcp.output_dims.size() - 1;
    if (harness_num_dims > SNIPPETS_MAX_HARNESS_DIMS) {
        canUseOptimizedImpl = false;
        harness_num_dims = SNIPPETS_MAX_HARNESS_DIMS;
    }
    for (size_t i = 0; i < inputShapes.size(); i++) {
        auto b = offsets_in[i].begin();
        std::copy(b, b + harness_num_dims, &jcp.data_offsets[i * harness_num_dims]);
    }
    for (size_t i = 0; i < outputShapes.size(); i++) {
        auto b = offsets_out[i].begin();
        std::copy(b, b + harness_num_dims, &jcp.data_offsets[(inputShapes.size() + i) * harness_num_dims]);
    }
    schedule = snippet->generate(reinterpret_cast<void*>(&jcp));
}

void Snippet::schedule_6d(const jit_snippets_call_args& call_args) const {
    const auto& dom = exec_domain;
    // < N, C, H, W > < 1, 1, N, C*H*W>
    parallel_for5d(dom[0], dom[1], dom[2], dom[3], dom[4],
        [&](int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4) {
            int64_t indexes[] = {d0, d1, d2, d3, d4};

#ifdef CPU_DEBUG_CAPS
            std::cout << "d0 = " << d0 << ", d1 = " << d1 << ", d2 = " << d2 << ", d3 = " << d3 << ", d4 = " << d4 << std::endl;
#endif

            auto callable = schedule.get_callable<kernel>();
            callable(indexes, &call_args);
        });
}

void Snippet::schedule_nt(const jit_snippets_call_args& call_args) const {
    const auto& work_size = exec_domain;
    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        splitter(schedulerWorkAmount, nthr, ithr, start, end);

        std::vector<int64_t> indexes(work_size.size() - 1, 0);
        for (size_t iwork = start; iwork < end; ++iwork) {
            size_t tmp = iwork;
            for (ptrdiff_t j = work_size.size() - 2; j >= 0; j--) {
                indexes[j] = tmp % work_size[j];
                tmp /= work_size[j];
            }

            auto callable = schedule.get_callable<kernel>();
            callable(indexes.data(), &call_args);
        }
    });
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
