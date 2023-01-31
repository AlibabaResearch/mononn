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

namespace mononn_engine {
namespace core {
namespace op {
class ClusterConv : public ClusterOp {
 public:
  ClusterConv(std::string _name, std::vector<std::shared_ptr<Op>> _operands,
              std::vector<TensorSpec> _output_specs)
      : ClusterOp(_name, _operands, _output_specs) {}

  TensorShape get_loop_shape() const override;
  Schedule construct_schedule(LocalityTier::Tier tier) override;

  void setup_codegen() override;
  std::string generate_cluster_code() const override;
  ClusterType get_cluster_type() const override;

 private:
};
}  // namespace op
}  // namespace core
}  // namespace mononn_engine