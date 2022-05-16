// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

/**
 * @interface ReplaceLoadsWithSplitScalarLoads
 * @brief Replaces vector loads with scalar versions.
 * The pass is used to cange element type of function in a canonical form vector to scalar.
 * Used for tail generation
 * @ingroup snippets
 */
class ReplaceLoadsWithSplitLoads: public ngraph::pass::MatcherPass {
public:
    ReplaceLoadsWithSplitLoads();
};

} // namespace pass
} // namespace snippets
} // namespace ngraph
