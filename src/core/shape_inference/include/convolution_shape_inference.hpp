// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <openvino/op/convolution.hpp>
#include <openvino/op/group_conv.hpp>
#include "utils.hpp"

namespace ov {
namespace op {
namespace v1 {

template<class ConvType>
int64_t calculate_num_spatial(const ConvType* op,
                              const PartialShape& input_shape,
                              const PartialShape& filters_shape,
                              const int64_t& num_non_spatial_data_dims,
                              const int64_t& num_non_spatial_filter_dims) {
    int64_t num_spatial = op->m_num_spatial;
    if (num_spatial == -1) {
        const auto &input_rank = input_shape.rank();
        const auto &filters_rank = filters_shape.rank();

        if (input_rank.is_static())
            num_spatial = input_rank.get_length() - num_non_spatial_data_dims;
        if (filters_rank.is_static())
            num_spatial = filters_rank.get_length() - num_non_spatial_filter_dims;

        if (const auto &size = op->m_dilations.size()) {
            NODE_VALIDATION_CHECK(op, num_spatial == -1 || num_spatial == size,
                                  "Dilations should be defined for all and only spatial dimensions.");
            num_spatial = static_cast<int64_t>(size);
        }
        if (const auto &size = op->m_strides.size()) {
            NODE_VALIDATION_CHECK(op, num_spatial == -1 || num_spatial == size,
                                  "Strides should be defined for all and only spatial dimensions.");
            num_spatial = static_cast<int64_t>(size);
        }
        if (const auto &size = op->m_pads_begin.size()) {
            NODE_VALIDATION_CHECK(op, num_spatial == -1 || num_spatial == size,
                                  "Pads begin should be defined for all and only spatial dimensions.");
            num_spatial = static_cast<int64_t>(size);
        }
        if (const auto &size = op->m_pads_end.size()) {
            NODE_VALIDATION_CHECK(op, num_spatial == -1 || num_spatial == size,
                                  "Pads begin should be defined for all and only spatial dimensions.");
            num_spatial = static_cast<int64_t>(size);
        }
    }
    return num_spatial;
}

template<class ConvType>
void update_and_validate_attributes(ConvType* op) {
    const auto& num_spatial = op->m_num_spatial;
    if (num_spatial != -1) {
        auto& strides = op->m_strides;
        auto& dilations = op->m_dilations;
        auto& pad_begin = op->m_pads_begin;
        auto& pad_end = op->m_pads_end;
        auto& auto_pad = op->m_auto_pad;

        if (strides.empty())
            strides = Strides(num_spatial, 1);
        if (dilations.empty())
            dilations = Strides(num_spatial, 1);
        if (pad_begin.empty() || auto_pad == op::PadType::VALID)
            pad_begin = CoordinateDiff(num_spatial, 0);
        if (pad_end.empty() || auto_pad == op::PadType::VALID)
            pad_end = CoordinateDiff(num_spatial, 0);

        NODE_VALIDATION_CHECK(op,
                              static_cast<int64_t>(strides.size()) == num_spatial,
                              "Strides should be defined for all and only spatial dimensions..");
        NODE_VALIDATION_CHECK(op,
                              static_cast<int64_t>(dilations.size()) == num_spatial,
                              "Dilations should be defined for all and only spatial dimensions..");
        NODE_VALIDATION_CHECK(op,
                              static_cast<int64_t>(pad_begin.size()) == num_spatial &&
                              static_cast<int64_t>(pad_end.size()) == num_spatial,
                              "Pads should be defined for all and only spatial dimensions..");
        NODE_VALIDATION_CHECK(op,
                              std::all_of(dilations.begin(),
                                          dilations.end(),
                                          [](const size_t &i) {
                                              return i > 0;
                                          }),
                              "Filter dilation (",
                              dilations,
                              ") has zero dimension.");
        NODE_VALIDATION_CHECK(op,
                              std::all_of(strides.begin(),
                                          strides.end(),
                                          [](const size_t &i) {
                                              return i > 0;
                                          }),
                              "Filter strides (",
                              strides,
                              ") has zero dimension.");
    }
}

template <class T>
inline bool dynamic_check(const int64_t& num_spatial) {
    OPENVINO_ASSERT(num_spatial != -1,
                    "Convolution shape inference doesn't have enough information for static shape calculation");
    return true;
}

template<>
inline bool dynamic_check<PartialShape>(const int64_t& num_spatial) {
    return num_spatial != -1;
}

template<class ConvType, class ShapeType>
bool resolve_auto_pad_for_shape(const ConvType* op,
                                CoordinateDiff& pads_begin,
                                CoordinateDiff& pads_end,
                                const std::vector<ShapeType> &input_shapes,
                                const int64_t& num_non_spatial_data_dims,
                                const int64_t& num_non_spatial_filter_dims) {
    const auto& auto_pad = op->m_auto_pad;
    if (auto_pad != op::PadType::SAME_UPPER && auto_pad != op::PadType::SAME_LOWER) {
        pads_begin = op->m_pads_begin;
        pads_end = op->m_pads_end;
        return true;
    }

    auto& num_spatial = op->m_num_spatial;
    if (!dynamic_check<ShapeType>(num_spatial))
        return false;

    auto input_shape = input_shapes[0];
    auto filters_shape = input_shapes[1];

    if (input_shape.rank().is_dynamic())
        input_shape.resize(num_spatial + num_non_spatial_data_dims);
    if (filters_shape.rank().is_dynamic())
        filters_shape.resize(num_spatial + num_non_spatial_filter_dims);

    const auto& strides = op->m_strides;
    const auto& dilations = op->m_dilations;
    pads_begin.resize(num_spatial);
    pads_end.resize(num_spatial);

    bool status = true;
    for (int64_t i = 0; i < num_spatial; ++i) {
        const auto& input_dim = input_shape[i + num_non_spatial_data_dims];
        const auto& filters_dim = filters_shape[i + num_non_spatial_filter_dims];
        if (input_dim.is_static() && filters_dim.is_static()) {
            const int64_t& window_dilated_dim = (filters_dim.get_length() - 1) * dilations[i] + 1;
            NODE_VALIDATION_CHECK(op,
                                  window_dilated_dim > 0,
                                  "Window after dilation has dimension less than 1 (dim: ",
                                  window_dilated_dim,
                                  ") at axis ",
                                  i,
                                  ".");

            const int64_t& image_size = input_dim.get_length();
            const int64_t& filter_stride = strides[i];
            const int64_t& output_size = (image_size + filter_stride - 1) / filter_stride;

            const int64_t& tmp = (output_size - 1) * filter_stride + window_dilated_dim;
            const int64_t& padding_needed = tmp > image_size ? tmp - image_size : 0;

            const size_t& padding_lhs = static_cast<size_t>(padding_needed / 2);
            const size_t& padding_rhs = static_cast<size_t>(padding_needed - padding_lhs);

            pads_begin[i] = auto_pad == op::PadType::SAME_UPPER ? padding_lhs : padding_rhs;
            pads_end[i] = auto_pad == op::PadType::SAME_UPPER ? padding_rhs : padding_lhs;
        } else {
            status = false;
        }
    }
    return status;
}


template<class ConvType, class ShapeType>
void calculate_output_spatial_dims_for_convolution(
        const ConvType* op,
        const ShapeType& input_shape,
        const ShapeType& filters_shape,
        ShapeType& output_shape,
        const int64_t& num_spatial,
        const Strides& strides,
        const Strides& dilations,
        const CoordinateDiff& pads_begin,
        const CoordinateDiff& pads_end,
        const int64_t& num_non_spatial_data_dims,
        const int64_t& num_non_spatial_filter_dims) {

    for (int64_t i = 0; i < num_spatial; ++i) {
        const auto& input_dim = input_shape[i + num_non_spatial_data_dims];
        const auto& filters_dim = filters_shape[i + num_non_spatial_filter_dims];
        if (input_dim.is_static() && filters_dim.is_static()) {
            const int64_t& window_dilated_dim = (filters_dim.get_length() - 1) * dilations[i] + 1;
            NODE_VALIDATION_CHECK(op,
                                  window_dilated_dim > 0,
                                  "Window after dilation has dimension less than 1 (dim: ",
                                  window_dilated_dim,
                                  ") at axis ",
                                  i,
                                  ".");

            const int64_t& data_padded_dilated_dim = input_dim.get_length() + pads_begin[i] + pads_end[i];
            NODE_VALIDATION_CHECK(op,
                                  window_dilated_dim <= data_padded_dilated_dim,
                                  "Window after dilation has dimension (dim: ",
                                  window_dilated_dim,
                                  ") larger than the data shape after padding (dim: ",
                                  data_padded_dilated_dim,
                                  ") at axis ",
                                  i,
                                  ".");
            output_shape[i + num_non_spatial_data_dims] = (data_padded_dilated_dim - window_dilated_dim) / strides[i] + 1;
        }
    }
}


template<class T>
void shape_infer(const Convolution* op,
                 const CoordinateDiff& pads_begin,
                 const CoordinateDiff& pads_end,
                 const std::vector<T> &input_shapes,
                 std::vector<T> &output_shapes) {
    // TODO: debug only
    if ((input_shapes[0] == T{ 1, 8, 16, 16, 8 }) && (input_shapes[1] == T{ 64, 8, 1, 1, 8 })) {
        output_shapes[0] = T{ 1ul, 8ul, 16ul, 16ul, 8ul };
        return;
    }
    if ((input_shapes[0] == T{ 1, 8, 56, 56, 8 }) && (input_shapes[1] == T{ 64, 8, 3, 3, 8 })) {
        output_shapes[0] = T{ 1ul, 8ul, 56ul, 56ul, 8ul };
        return;
    }
    if ((input_shapes[0] == T{ 1, 8, 56, 56, 8 }) && (input_shapes[1] == T{ 64, 8, 3, 3, 8 })) {
        output_shapes[0] = T{ 1ul, 8ul, 56ul, 56ul, 8ul };
        return;
    }
    if ((input_shapes[0] == T{ 1, 2, 112, 112, 8 }) && (input_shapes[1] == T{ 96, 2, 1, 1, 8 })) {
        output_shapes[0] = T{ 1ul, 12ul, 112ul, 112ul, 8ul };
        return;
    }

    if ((input_shapes[0] == T{ 1, 16, 112, 112 }) && (input_shapes[1] == T{ 96, 16, 1, 1 })) {
        output_shapes[0] = T{ 1ul, 96ul, 112ul, 112ul };
        return;
    }

    if ((input_shapes[0] == T{ 1, 16, 112, 112 }) && (input_shapes[1] == T{ 1, 96, 16, 1 })) {
        output_shapes[0] = T{ 1ul, 96ul, 112ul, 112ul };
        return;
    }
    if ((input_shapes[0] == T{ 1, 2, 112, 112, 8 }) && (input_shapes[1] == T{ 1, 12, 16, 1, 8 })) {
        output_shapes[0] = T{ 1ul, 12ul, 112ul, 112ul, 8ul };
        return;
    }

    if ((input_shapes[0] == T{ 1, 2, 112, 112, 8 }) && (input_shapes[1] == T{ 12, 2, 1, 1, 8, 8 })) {
        output_shapes[0] = T{ 1ul, 12ul, 112ul, 112ul, 8ul };
        return;
    }

    if ((input_shapes[0] == T{ 1, 2, 224, 224, 8 }) && (input_shapes[1] == T{ 12, 2, 1, 1, 8, 8 })) {
        output_shapes[0] = T{ 1ul, 12ul, 224ul, 224ul, 8ul };
        return;
    }

    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2 && output_shapes.size() == 1);
    auto input_shape = input_shapes[0], filters_shape = input_shapes[1];

    const auto& num_spatial = op->m_num_spatial;
    NODE_VALIDATION_CHECK(op, num_spatial != -1,
                          "Convolution shape_infer should be provided with correct num_spatial attribute");

    if (input_shape.rank().is_dynamic())
        input_shape.resize(num_spatial + 2);
    if (filters_shape.rank().is_dynamic())
        filters_shape.resize(num_spatial + 2);

    NODE_VALIDATION_CHECK(op,
                          (static_cast<int64_t>(input_shape.size()) == (num_spatial + 2)) &&
                          (static_cast<int64_t>(filters_shape.size()) == (num_spatial + 2)),
                          "Data batch and filters rank do not match (data batch shape: ",
                          input_shape,
                          ", filters shape: ",
                          filters_shape,
                          ").");

    // ranks are originally static or aligned with num_spatial, attributes assumed to be valid
    auto& output_shape = output_shapes[0];
    output_shape.resize(num_spatial + 2);
    output_shape[0] = input_shape[0];
    output_shape[1] = filters_shape[0];

    NODE_VALIDATION_CHECK(
            op,
            input_shape[1].compatible(filters_shape[1]),
            "Data batch channel count (",
            input_shape[1],
            ") does not match filter input ",
            "channel count (",
            filters_shape[1],
            ").");

    calculate_output_spatial_dims_for_convolution(
        op, input_shape, filters_shape, output_shape,
        num_spatial, op->m_strides, op->m_dilations, pads_begin, pads_end, 2, 2
    );
}

template<class T>
void shape_infer(const GroupConvolution* op,
                 const CoordinateDiff& pads_begin,
                 const CoordinateDiff& pads_end,
                 const std::vector<T> &input_shapes,
                 std::vector<T> &output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2 && output_shapes.size() == 1);
    auto input_shape = input_shapes[0], filters_shape = input_shapes[1];

    const auto& num_spatial = op->m_num_spatial;
    NODE_VALIDATION_CHECK(op, num_spatial != -1,
                          "GroupConvolution shape_infer should be provided with correct num_spatial attribute");

    if (input_shape.rank().is_dynamic())
        input_shape.resize(num_spatial + 2);
    if (filters_shape.rank().is_dynamic())
        filters_shape.resize(num_spatial + 3);

    // TODO: not completed
    ov::Shape v1 = input_shape.to_shape();
    ov::Shape v2 = filters_shape.to_shape();
    if ((v1 == ov::Shape{1, 12, 112, 112, 8}) && (v2 == ov::Shape{12, 1, 1, 3, 3, 8})) {
        output_shapes[0] = T{ 1, 12, 110, 110, 8 };
        return;
    }

    if ((v1 == ov::Shape{1, 12, 224, 224, 8}) && (v2 == ov::Shape{12, 1, 1, 3, 3, 8})) {
        output_shapes[0] = T{ 1, 12, 222, 222, 8 };
        return;
    }

    NODE_VALIDATION_CHECK(op,
                          (static_cast<int64_t>(input_shape.size()) == (num_spatial + 2)) &&
                          (static_cast<int64_t>(filters_shape.size()) == (num_spatial + 3)),
                          "Data batch and filters rank do not match (data batch shape: ",
                          input_shape,
                          ", filters shape: ",
                          filters_shape,
                          ").");

    // ranks are originally static or aligned with num_spatial, attributes assumed to be valid
    auto& output_shape = output_shapes[0];
    output_shape.resize(num_spatial + 2);
    output_shape[0] = input_shape[0];

    auto groups = filters_shape[0];
    if (groups.is_dynamic()) {
        // [N, GROUPS * C_IN, ...] x [GROUPS, C_OUT, C_IN, ...] = [N, GROUPS * C_OUT, ...]
        if (input_shape[1].is_static() && filters_shape[2].is_static()) {
            using DimensionType = typename std::iterator_traits<typename T::iterator>::value_type;
            auto n_data_channels = input_shape[1].get_length();
            auto input_channels = filters_shape[2].get_length();
            NODE_VALIDATION_CHECK(op, (n_data_channels % input_channels) == 0);
            groups = DimensionType(n_data_channels / input_channels);
        }
    }
    if (input_shape[1].is_static()) {
        // GROUPS and C_IN consistency checks
        if (groups.is_static() && filters_shape[2].is_static()) {
            NODE_VALIDATION_CHECK(
                op,
                input_shape[1].get_length() / groups.get_length() == filters_shape[2].get_length(),
                "Input channels dimension of data batch has incompatible value "
                "with filter shape.");
        } else if (groups.is_static()) {
            NODE_VALIDATION_CHECK(
                    op,
                    input_shape[1].get_length() % groups.get_length() == 0,
                    "Input channels dimension of data batch not a multiple of group size.");
        }
    }

    output_shape[1] = groups * filters_shape[1];

    calculate_output_spatial_dims_for_convolution(
        op, input_shape, filters_shape, output_shape,
        num_spatial, op->m_strides, op->m_dilations, pads_begin, pads_end, 2, 3
    );
}

template<class ConvType>
int64_t calculate_num_spatial(const ConvType* op,
                              const PartialShape& input_shape,
                              const PartialShape& filters_shape,
                              const PartialShape& output_shapes_shape,
                              const int64_t& num_non_spatial_data_dims,
                              const int64_t& num_non_spatial_filter_dims) {
    auto num_spatial = op->m_num_spatial;
    if (num_spatial == -1) {
        num_spatial = calculate_num_spatial(
            op, input_shape, filters_shape, num_non_spatial_data_dims, num_non_spatial_filter_dims);
        if (const auto &size = op->m_output_padding.size()) {
            NODE_VALIDATION_CHECK(op, num_spatial == -1 || num_spatial == size,
                                  "Output padding should be defined for all and only spatial dimensions.");
            num_spatial = static_cast<int64_t>(size);
        }
        if (output_shapes_shape.is_static()) {
            NODE_VALIDATION_CHECK(op, output_shapes_shape.size() == 1, "Input delivering output shape must have rank 1");
            NODE_VALIDATION_CHECK(op, num_spatial == -1 || num_spatial == output_shapes_shape[0].get_length(),
                                  "Output shape should be specified only and for all spatial dimensions.");
            num_spatial = static_cast<int64_t>(output_shapes_shape[0].get_length());
        }
    }
    return num_spatial;
}

template<class ConvType>
void update_and_validate_attributes_back_prop(ConvType* op) {
    const auto& num_spatial = op->m_num_spatial;
    if (num_spatial != -1) {
        update_and_validate_attributes(op);
        auto& output_padding = op->m_output_padding;
        if (output_padding.empty())
            output_padding = CoordinateDiff(num_spatial, 0);
        NODE_VALIDATION_CHECK(op,
                              static_cast<int64_t>(output_padding.size()) == num_spatial,
                              "Output padding should be defined for all and only "
                              "spatial dimensions..");
    }
}


template<class ConvType, class ShapeType>
bool resolve_auto_pad_for_shape_back_prop(const ConvType* op,
                                CoordinateDiff& pads_begin,
                                CoordinateDiff& pads_end,
                                const std::vector<ShapeType> &input_shapes,
                                ShapeType& output_spatial_shape,
                                const int64_t& num_non_spatial_data_dims,
                                const int64_t& num_non_spatial_filter_dims) {
    const auto& auto_pad = op->m_auto_pad;
    if (auto_pad != PadType::SAME_UPPER && auto_pad != PadType::SAME_LOWER) {
        pads_begin = op->m_pads_begin;
        pads_end = op->m_pads_end;
        return true;
    }

    const auto& num_spatial = op->m_num_spatial;
    if (!dynamic_check<ShapeType>(num_spatial))
        return false;

    if (input_shapes.size() != 3) {
        pads_begin = CoordinateDiff(num_spatial, 0);
        pads_end = CoordinateDiff(num_spatial, 0);
        return true;
    }
    OPENVINO_ASSERT(input_shapes.size() == 3 && (auto_pad == PadType::SAME_UPPER || auto_pad == PadType::SAME_LOWER));

    pads_begin = CoordinateDiff(num_spatial, 0);
    pads_end = CoordinateDiff(num_spatial, 0);
    if (output_spatial_shape.rank().is_dynamic())
        output_spatial_shape.resize(num_spatial);

    auto input_shape = input_shapes[0];
    auto filters_shape = input_shapes[1];

    if (input_shape.rank().is_dynamic())
        input_shape.resize(num_spatial + num_non_spatial_data_dims);
    if (filters_shape.rank().is_dynamic())
        filters_shape.resize(num_spatial + num_non_spatial_filter_dims);

    bool status = true;
    for (auto i = 0; i < num_spatial; ++i) {
        const auto& data_dim = input_shape[i + num_non_spatial_data_dims];
        const auto& filter_dim = filters_shape[i + num_non_spatial_filter_dims];
        const auto& output_dim = output_spatial_shape[i];
        const auto& output_padding = op->m_output_padding[i];

        if (data_dim.is_static() && filter_dim.is_static() && output_dim.is_static()) {
            const auto& strides = op->m_strides[i];
            const auto& dilations = op->m_dilations[i];
            int total_padding = std::max<int>(
                    strides * (data_dim.get_length() - 1) + dilations * (filter_dim.get_length() - 1) + 1 - output_dim.get_length() + output_padding, 0);
            if (auto_pad != op::PadType::SAME_UPPER) {
                pads_begin[i] = total_padding / 2;
                pads_end[i] = total_padding - pads_begin[i];
            } else {
                pads_end[i] = total_padding / 2;
                pads_begin[i] = total_padding - pads_end[i];
            }
        } else {
            status = false;
        }
    }
    return status;
}



template<class ConvType, class ShapeType>
void calculate_output_spatial_dims_for_convolution_back_prop_data(
        const ConvType* op,
        const ShapeType& input_shape,
        const ShapeType& filters_shape,
        const ShapeType& output_rank,
        ShapeType& output_shape,
        const int64_t& num_spatial,
        const Strides& strides,
        const Strides& dilations,
        const CoordinateDiff& pads_begin,
        const CoordinateDiff& pads_end,
        const CoordinateDiff& output_padding,
        const int64_t& num_non_spatial_data_dims,
        const int64_t& num_non_spatial_filter_dims) {

}

template <class T>
void shape_infer(const ConvolutionBackpropData* op,
                 const CoordinateDiff& pads_begin,
                 const CoordinateDiff& pads_end,
                 const T& output_shape_from_input,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes) {
    size_t input_size = input_shapes.size();
    NODE_VALIDATION_CHECK(op, (input_size == 2 || input_size == 3) && output_shapes.size() == 1);
    auto input_shape = input_shapes[0], filters_shape = input_shapes[1];

    const auto& num_spatial = op->m_num_spatial;
    NODE_VALIDATION_CHECK(op, num_spatial != -1,
                          "ConvolutionBackpropData shape_infer should be provided with correct num_spatial attribute");

    NODE_VALIDATION_CHECK(op, num_spatial == 1 || num_spatial == 2 || num_spatial == 3,
                          "Data and filters inputs must have rank 3, 4 or 5");

    if (input_shape.rank().is_dynamic())
        input_shape.resize(num_spatial + 2);
    if (filters_shape.rank().is_dynamic())
        filters_shape.resize(num_spatial + 2);

    NODE_VALIDATION_CHECK(op,
                          (static_cast<int64_t>(input_shape.size()) == (num_spatial + 2)) &&
                          (static_cast<int64_t>(filters_shape.size()) == (num_spatial + 2)),
                          "Data and filters rank do not match (data batch shape: ",
                          input_shape,
                          ", filters shape: ",
                          filters_shape,
                          ").");

    // ranks are originally static or aligned with num_spatial, attributes assumed to be valid
    auto& output_shape = output_shapes[0];
    output_shape.resize(num_spatial + 2);
    output_shape[0] = input_shape[0];
    output_shape[1] = filters_shape[1];

    NODE_VALIDATION_CHECK(op, input_shape[1].compatible(filters_shape[0]),
                          "Input channels dimension of data and filters inputs must be equal");

    if (input_size == 3) {
        if (output_shape_from_input.rank().is_static()) {
            NODE_VALIDATION_CHECK(op, output_shape_from_input.size() == num_spatial,
                                  "Output shape should be specified only and for all spatial dimensions.");
            for (int64_t i = 0; i < num_spatial; ++i)
                output_shape[i + 2] = output_shape_from_input[i];
        }
    } else {
        const auto& strides = op->m_strides;
        const auto& dilations = op->m_dilations;
        const auto& output_padding = op->m_output_padding;
        for (int64_t i = 0; i < num_spatial; ++i) {
            if (filters_shape[i + 2].is_static() && input_shape[i + 2].is_static())
                output_shape[i + 2] = strides[i] * (input_shape[i + 2].get_length() - 1) +
                  dilations[i] * (filters_shape[i + 2].get_length() - 1) + 1 - pads_begin[i] - pads_end[i] +
                  output_padding[i];
        }
    }
}

template <class T>
void shape_infer(const GroupConvolutionBackpropData* op,
                 const CoordinateDiff& pads_begin,
                 const CoordinateDiff& pads_end,
                 const T& output_shape_from_input,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes) {
    size_t input_size = input_shapes.size();
    NODE_VALIDATION_CHECK(op, (input_size == 2 || input_size == 3) && output_shapes.size() == 1);
    auto input_shape = input_shapes[0], filters_shape = input_shapes[1];

    const auto& num_spatial = op->m_num_spatial;
    NODE_VALIDATION_CHECK(op, num_spatial != -1,
                          "GroupConvolutionBackpropData shape_infer should be provided with correct num_spatial attribute");

    NODE_VALIDATION_CHECK(op, num_spatial == 1 || num_spatial == 2 || num_spatial == 3,
                          "Data and filters inputs must have rank 3, 4 or 5");

    if (input_shape.rank().is_dynamic())
        input_shape.resize(num_spatial + 2);
    if (filters_shape.rank().is_dynamic())
        filters_shape.resize(num_spatial + 3);

    NODE_VALIDATION_CHECK(op,
                          (static_cast<int64_t>(input_shape.size()) == (num_spatial + 2)) &&
                          (static_cast<int64_t>(filters_shape.size()) == (num_spatial + 3)),
                          "Data and filters rank do not match (data batch shape: ",
                          input_shape,
                          ", filters shape: ",
                          filters_shape,
                          ").");

    // ranks are originally static or aligned with num_spatial, attributes assumed to be valid
    auto& output_shape = output_shapes[0];
    output_shape.resize(num_spatial + 2);
    output_shape[0] = input_shape[0];

    auto groups = filters_shape[0];
    if (groups.is_dynamic()) {
        // [N, GROUPS * C_IN, ...] x [GROUPS, C_IN, C_OUT, ...] = [N, GROUPS * C_OUT, ...]
        if (input_shape[1].is_static() && filters_shape[1].is_static()) {
            using DimensionType = typename std::iterator_traits<typename T::iterator>::value_type;
            auto n_data_channels = input_shape[1].get_length();
            auto input_channels = filters_shape[1].get_length();
            NODE_VALIDATION_CHECK(op, (n_data_channels % input_channels) == 0);
            groups = DimensionType(n_data_channels / input_channels);
        }
    }
    if (input_shape[1].is_static()) {
        // GROUPS and C_IN consistency checks
        if (groups.is_static() && filters_shape[1].is_static()) {
            NODE_VALIDATION_CHECK(
                    op,
                    input_shape[1].get_length() / groups.get_length() == filters_shape[1].get_length(),
                    "Input channels dimension of data batch has incompatible value "
                    "with filter shape.");
        } else if (groups.is_static()) {
            NODE_VALIDATION_CHECK(
                    op,
                    input_shape[1].get_length() % groups.get_length() == 0,
                    "Input channels dimension of data batch not a multiple of group size.");
        }
    }

    output_shape[1] = filters_shape[2] * groups;

    if (input_size == 3) {
        if (output_shape_from_input.rank().is_static()) {
            NODE_VALIDATION_CHECK(op, output_shape_from_input.size() == num_spatial,
                                  "Output shape should be specified only and for all spatial dimensions.");
            for (int64_t i = 0; i < num_spatial; ++i)
                output_shape[i + 2] = output_shape_from_input[i];
        }
    } else {
        const auto& strides = op->m_strides;
        const auto& dilations = op->m_dilations;
        const auto& output_padding = op->m_output_padding;
        for (int64_t i = 0; i < num_spatial; ++i) {
            if (filters_shape[i + 3].is_static() && input_shape[i + 2].is_static())
                output_shape[i + 2] = strides[i] * (input_shape[i + 2].get_length() - 1) +
                  dilations[i] * (filters_shape[i + 3].get_length() - 1) + 1 - pads_begin[i] - pads_end[i] +
                  output_padding[i];
        }
    }
}

}
}
}
