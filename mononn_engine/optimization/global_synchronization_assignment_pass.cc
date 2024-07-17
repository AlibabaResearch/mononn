// Copyright 2023 The MonoNN Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mononn_engine/optimization/global_synchronization_assignment_pass.h"

#include "mononn_engine/core/gpu/synchronization.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/optimization/common.h"

namespace mononn_engine {
namespace optimization {
using OpType = mononn_engine::core::op::OpType;
using Synchronization = mononn_engine::core::gpu::Synchronization;

std::string GlobalSynchronizationAssignmentPass::name() const {
  return PassName::GlobalSynchronizationAssignmentPass;
}

bool GlobalSynchronizationAssignmentPass::run(
    Graph* graph, std::shared_ptr<CUDAContext> cuda_context) {
  for (auto const& node_name : graph->get_node_list()) {
    for (auto& edge : graph->get_node_output_edges(node_name)) {
      if (edge->get_dst()->get_type() == OpType::get_tuple_element) {
        edge->set_sync(Synchronization::None);
        continue;
      }

      if (edge->get_src()->get_type() == OpType::parameter ||
          edge->get_src()->get_type() == OpType::constant ||
          edge->get_src()->get_type() == OpType::iota) {
        edge->set_sync(Synchronization::None);
      } else {
        edge->set_sync(Synchronization::Global);
      }
    }
  }

  return true;
}
}  // namespace optimization
}  // namespace mononn_engine
