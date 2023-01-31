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
#include "mononn_engine/core/op_impl/parameter_impl_base.h"
#include "mononn_engine/core/tensor/tensor.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
class ParameterImpl : public ParameterImplBase {
 public:
  using CUDAContext = mononn_engine::core::context::CUDAContext;
  using Tensor = mononn_engine::core::tensor::Tensor;
  using Dtype = mononn_engine::core::tensor::Dtype;

  ParameterImpl(std::shared_ptr<CUDAContext> _cuda_context, Tensor _output)
      : cuda_context(_cuda_context), output(_output) {}

  std::string generate_with_index_impl() const override;

  std::vector<Tensor> get_input_tensor() const override;
  std::vector<Tensor> get_output_tensor() const override;
  int get_elements_per_access() const override;
  void set_instruction_parallel_factor(int _ilp_factor) override;

  static std::vector<std::shared_ptr<OpImplBase>> get_available_implementations(
      std::shared_ptr<CUDAContext> cuda_context, Tensor output);

 private:
  std::string memory_read_wrapper(Dtype access_type, std::string var_name,
                                  std::string src_ptr, std::string offset,
                                  bool define_variable, std::string pred) const;
  std::shared_ptr<CUDAContext> cuda_context;
  Tensor output;
};
}  // namespace op_impl
}  // namespace core
}  // namespace mononn_engine