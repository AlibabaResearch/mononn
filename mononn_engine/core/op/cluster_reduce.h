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
#include <memory>
#include <vector>

#include "mononn_engine/core/op/cluster_op.h"

namespace mononn_engine {
namespace core {
namespace op {
class ClusterReduce : public ClusterOp {
 public:
  ClusterReduce(std::string _name, std::vector<std::shared_ptr<Op>> _operands,
                std::vector<TensorSpec> _output_specs);

  TensorShape get_loop_shape() const override;
  Schedule construct_schedule(LocalityTier::Tier tier) override;

  void setup_codegen() override;
  std::string generate_cluster_code() const override;

  std::vector<Op*> get_reduce_nodes();
  std::vector<const Op*> get_reduce_nodes() const;

  std::vector<Op*> get_reduce_nodes_in_last_sub_cluster();
  std::vector<const Op*> get_reduce_nodes_in_last_sub_cluster() const;

  void trace_symbolic_index() override;

  bool is_cluster_reduce() const override;
  ClusterType get_cluster_type() const override;

  int get_reduction_dimension_size() const;

  std::string generate_async_pipeline_initialization() const override;
  std::string generate_async_pipeline_prefetch() const override;

  void initialize_smem_manager() override;

 private:
};
}  // namespace op
}  // namespace core
}  // namespace mononn_engine