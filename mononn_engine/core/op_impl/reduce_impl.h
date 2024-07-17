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
#include <unordered_map>
#include <vector>

#include "mononn_engine/codegen/reduction_functor_generator.h"
#include "mononn_engine/core/context/cuda_context.h"
#include "mononn_engine/core/gpu/functor.h"
#include "mononn_engine/core/op_annotation/locality_tier.h"
#include "mononn_engine/core/op_impl/op_impl_base.h"
#include "mononn_engine/core/op_impl/reducer.h"
#include "mononn_engine/core/semantic/function_invocation.h"
#include "mononn_engine/core/tensor/dtype.h"
#include "mononn_engine/core/tensor/scalar.h"
#include "mononn_engine/core/tensor/tensor.h"
#include "mononn_engine/core/tensor/tensor_spec.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
class ReduceImpl : public OpImplBase {
 public:
  using Scalar = mononn_engine::core::tensor::Scalar;
  using TensorSpec = mononn_engine::core::tensor::TensorSpec;
  using Tier = mononn_engine::core::op_annotation::LocalityTier::Tier;
  using LocalityTier = mononn_engine::core::op_annotation::LocalityTier;
  using Functor = mononn_engine::core::gpu::Functor;
  using FunctionInvocation = mononn_engine::core::semantic::FunctionInvocation;
  using CUDAContext = mononn_engine::core::context::CUDAContext;
  using Tensor = mononn_engine::core::tensor::Tensor;
  using Dtype = mononn_engine::core::tensor::Dtype;
  using ReductionFunctorGenerator =
      mononn_engine::codegen::ReductionFunctorGenerator;

  struct InputSpec {
    std::vector<Tensor> operands;
    Scalar init_value;
    // std::string init_value;
    // Reducer reducer;
    int dimension;
    Tier tier;
    const ReductionFunctorGenerator* reduction_functor_generator;
    Scalar reduce_accum;
  };

  ReduceImpl(std::shared_ptr<CUDAContext> _context,
             ReduceImpl::InputSpec _input_spec, Tensor _output);

  Tier get_tier() const;
  // std::vector<FunctionInvocation> get_invocations() const override;
  std::string generate_impl() const override;
  std::string generate_reduce() const;
  std::vector<Tensor> get_input_tensor() const override;
  std::vector<Tensor> get_output_tensor() const override;
  int get_elements_per_access() const override;
  const Scalar& get_reduce_accum() const override;
  std::string get_post_reduce_if_statement() const;
  std::string get_post_reduce_if_end() const;
  int get_smem_usage_in_bytes() const override;
  void set_instruction_parallel_factor(int _ilp_factor) override;

  static std::vector<std::shared_ptr<OpImplBase>> get_available_implementations(
      std::shared_ptr<CUDAContext> cuda_context, InputSpec input_spec,
      Tensor output);

  static std::string get_prerequisite_definition();

 private:
  InputSpec input_spec;
  int elements_per_access;
  Tier tier;
  // Functor reducer;
  std::shared_ptr<CUDAContext> context;
  Tensor output;
  Scalar reduce_accum;

  std::string post_reduce_if_statement;

  // static Functor get_reduce_functor(Reducer reducer, Dtype element_type);
};
}  // namespace op_impl
}  // namespace core
}  // namespace mononn_engine