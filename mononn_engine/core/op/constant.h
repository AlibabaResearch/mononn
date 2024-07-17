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
class Constant : public Op {
 public:
  using TensorSpec = mononn_engine::core::tensor::TensorSpec;
  using Tier = mononn_engine::core::op_annotation::LocalityTier::Tier;

  Constant(std::string _name, std::vector<std::shared_ptr<Op>> _operands,
           std::vector<TensorSpec> _output_specs)
      : Op(_name, _operands, _output_specs) {}
  OpType get_type() const override;
  std::vector<std::shared_ptr<OpImpl>> generate_candidate_implementation(
      std::shared_ptr<CUDAContext> context, Tier tier) const override;

  // for scalar constant
  void set_value(std::string _value) override;
  std::string get_value() const override;

  bool is_scalar() const;

  void set_data_float(std::vector<float> const& _data_float);
  void set_data_half(std::vector<Eigen::half> const& _data_half);

  std::vector<float> const& get_data_float() const;
  std::vector<Eigen::half> const& get_data_half() const;

 private:
  std::string value;
  std::vector<float> data_float;
  std::vector<Eigen::half> data_half;
};
}  // namespace op
}  // namespace core
}  // namespace mononn_engine