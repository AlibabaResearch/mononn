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
#include <string>
#include <vector>

#include "mononn_engine/core/op/op.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/tensor/tensor_spec.h"

namespace mononn_engine {
namespace core {
namespace op {
class Broadcast : public Op {
 public:
  using TensorSpec = mononn_engine::core::tensor::TensorSpec;
  using Tier = mononn_engine::core::op_annotation::LocalityTier::Tier;

  Broadcast(std::string _name, std::vector<std::shared_ptr<Op>> _operands,
            std::vector<TensorSpec> _output_specs)
      : Op(_name, _operands, _output_specs) {}
  OpType get_type() const override;
  std::vector<std::shared_ptr<OpImpl>> generate_candidate_implementation(
      std::shared_ptr<CUDAContext> context, Tier tier) const override;

  void set_dimensions(std::vector<int> _dimensions);
  std::vector<int> get_dimensions() const;

 protected:
 private:
  // Indicate which dimension in the broadcast result from broadcast's operand
  // A empty dimension means scalar to multi dimentional tensor broadcast thus
  // no dimension in result is originated from broadcast's operand E.g., [512,
  // 768] <- [768], dimensions={1} [512, 768] <- [1], dimensions = {}
  std::vector<int> dimensions;
};
}  // namespace op
}  // namespace core
}  // namespace mononn_engine