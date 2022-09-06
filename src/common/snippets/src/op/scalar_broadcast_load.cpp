// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/scalar_broadcast_load.hpp"

using namespace ngraph;

snippets::op::ScalarBroadcastLoad::ScalarBroadcastLoad(const Output<Node>& x) : Load(x) {
}
