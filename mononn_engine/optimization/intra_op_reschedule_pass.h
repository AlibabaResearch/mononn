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

#pragma once

#include "mononn_engine/core/op/cluster_op.h"
#include "mononn_engine/helpers/helpers.h"
#include "mononn_engine/optimization/graph_pass.h"

namespace mononn_engine {
namespace optimization {
using ClusterOp = mononn_engine::core::op::ClusterOp;

class IntraOpReschedulePass : public GraphPass {
 public:
  IntraOpReschedulePass(std::string _cluster_node_name, int _ilp_factor)
      : cluster_node_name(_cluster_node_name), ilp_factor(_ilp_factor) {}

  std::string name() const override;
  bool run(Graph* graph, std::shared_ptr<CUDAContext> cuda_context) override;

  static bool can_rescheduled_with_ilp_factor(
      std::shared_ptr<CUDAContext> cuda_context, ClusterOp* cluster_node,
      int ilp_factor);

 private:
  bool run_for_elewise_cluster(Graph* graph,
                               std::shared_ptr<CUDAContext> cuda_context,
                               ClusterOp* cluster_node);
  bool run_for_reduce_cluster(Graph* graph,
                              std::shared_ptr<CUDAContext> cuda_context,
                              ClusterOp* cluster_node);

  std::string cluster_node_name;
  int ilp_factor;
};
}  // namespace optimization
}  // namespace mononn_engine
