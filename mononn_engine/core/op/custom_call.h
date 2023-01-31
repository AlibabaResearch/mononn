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
class CustomCall : public Op {
 public:
  using TensorSpec = mononn_engine::core::tensor::TensorSpec;
  using Tier = mononn_engine::core::op_annotation::LocalityTier::Tier;

  struct Target {
    static std::string const cublas_gemm;
    static std::string const cudnn_conv_forward;
  };

  CustomCall(std::string _name, std::vector<std::shared_ptr<Op>> _operands,
             std::vector<TensorSpec> _output_specs)
      : Op(_name, _operands, _output_specs) {}

  OpType get_type() const override;
  std::vector<std::shared_ptr<OpImpl>> generate_candidate_implementation(
      std::shared_ptr<CUDAContext> context, Tier tier) const override;

  void set_custom_call_target(std::string _custom_call_target);
  std::string get_custom_call_target() const;

  void set_backend_config_str(std::string str);
  std::string get_backend_config_str() const;

  bool is_gemm() const override;
  bool is_conv() const override;

  void set_filter_size(const std::vector<int>& _filter_size);
  const std::vector<int>& get_filter_size() const;

  void set_filter_stride(const std::vector<int>& _filter_stride);
  const std::vector<int>& get_filter_stride() const;

  void set_padding_low(const std::vector<int>& _padding_low);
  const std::vector<int>& get_padding_low() const;

  void set_padding_high(const std::vector<int>& _padding_high);
  const std::vector<int>& get_padding_high() const;

  void set_conv_output_GTE_node_name(const std::string& node_name);
  const std::string& get_conv_output_GTE_node_name() const;

 private:
  std::string custom_call_target;
  std::string backend_config_str;

  std::vector<int> filter_size;
  std::vector<int> filter_stride;
  std::vector<int> padding_low;
  std::vector<int> padding_high;

  std::string conv_output_GET_node_name;
};
}  // namespace op
}  // namespace core
}  // namespace mononn_engine