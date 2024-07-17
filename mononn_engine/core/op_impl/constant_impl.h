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

#include "mononn_engine/core/context/cuda_context.h"
#include "mononn_engine/core/op_impl/op_impl_base.h"
#include "mononn_engine/core/tensor/tensor.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
class ConstantImpl : public OpImplBase {
 public:
  using CUDAContext = mononn_engine::core::context::CUDAContext;
  using Tensor = mononn_engine::core::tensor::Tensor;

  struct InputSpec {
    std::string value;
  };

  ConstantImpl(std::shared_ptr<CUDAContext> _cuda_context,
               InputSpec _input_spec, Tensor _output)
      : cuda_context(_cuda_context), input_spec(_input_spec), output(_output) {}

  std::string generate_impl() const override;
  std::string generate_with_index_impl() const override;

  std::vector<Tensor> get_input_tensor() const override;
  std::vector<Tensor> get_output_tensor() const override;
  int get_elements_per_access() const override;
  void set_instruction_parallel_factor(int _ilp_factor) override;

  static std::vector<std::shared_ptr<OpImplBase>> get_available_implementations(
      std::shared_ptr<CUDAContext> cuda_context, InputSpec input_spec,
      Tensor output);

 private:
  std::shared_ptr<CUDAContext> cuda_context;
  InputSpec input_spec;
  Tensor output;
};
}  // namespace op_impl
}  // namespace core
}  // namespace mononn_engine