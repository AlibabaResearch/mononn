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

#include "mononn_engine/core/context/cuda_context.h"
#include "mononn_engine/core/op_impl/parameter_impl_base.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
using CUDAContext = mononn_engine::core::context::CUDAContext;

class SmemPrefetchImpl : public ParameterImplBase {
 public:
  using Tensor = mononn_engine::core::tensor::Tensor;
  using CUDAContext = mononn_engine::core::context::CUDAContext;
  using LocalityTier = mononn_engine::core::op_annotation::LocalityTier;
  using Dtype = mononn_engine::core::tensor::Dtype;

  struct InputSpec {
    LocalityTier::Tier tier;
  };

  SmemPrefetchImpl(std::shared_ptr<CUDAContext> _cuda_context,
                   InputSpec _input_spec, Tensor _output)
      : cuda_context(_cuda_context), input_spec(_input_spec), output(_output) {
    this->set_need_generate_with_index(true);  // this is temporary hack
  }

  std::string generate_with_index_impl() const override;
  int get_elements_per_access() const override;
  std::vector<Tensor> get_input_tensor() const override;
  std::vector<Tensor> get_output_tensor() const override;

  void set_instruction_parallel_factor(int _ilp_factor) override;

  std::string generate_async_pipeline_initialization() const;
  std::string generate_async_pipeline_prefetch() const;

  static std::vector<std::shared_ptr<OpImplBase>> get_available_implementations(
      std::shared_ptr<CUDAContext> cuda_context, InputSpec input_spec,
      Tensor output);

 protected:
  void instantiate_concrete_index_impl(
      const std::vector<SymbolicIndexStamp>& symbolic_index_list,
      const std::map<std::string, std::string>& params,
      const std::string& loop_stride) override;
  void instantiate_ilp_concrete_index_impl(
      const std::vector<SymbolicIndexStamp>& symbolic_index_list,
      const std::map<std::string, std::string>& params,
      const std::string& loop_stride, const std::string& ilp_stride) override;

  // void propagate_attributes_impl(const std::unordered_map<std::string,
  // std::string> &attrs) override;
 private:
  std::shared_ptr<CUDAContext> cuda_context;
  InputSpec input_spec;
  Tensor output;

  std::vector<ConcreteIndexStamp>
      async_pipeline_initialization_concrete_index_list;
  std::vector<ConcreteIndexStamp> async_pipeline_prefetch_concrete_index_list;
  std::vector<ConcreteIndexStamp> smem_access_concrete_index_list;
  std::vector<std::vector<ConcreteIndexStamp>>
      ilp_async_pipeline_initialization_concrete_index_list;
  std::vector<std::vector<ConcreteIndexStamp>>
      ilp_async_pipeline_prefetch_concrete_index_list;
  std::vector<std::vector<ConcreteIndexStamp>>
      ilp_smem_access_concrete_index_list;

  std::string async_pipeline_initialization_pred;
  // std::vector<std::string> ilp_async_pipeline_initialization_pred;

  std::string async_pipeline_prefetch_pred;
  // std::vector<std::string> ilp_async_pipeline_prefetch_pred;

  std::string access_smem_buf(const std::string& smem_buf_name,
                              const std::string& stage_id,
                              const std::string& index, const std::string& pred,
                              const std::string& default_value = "0") const;
  std::string access_global_mem_buf(const std::string& buf_name,
                                    const std::string& index,
                                    const Dtype& type) const;

  std::string generate_async_copy_invocation(int bytes_per_access,
                                             std::string smem_ptr,
                                             std::string global_mem_ptr,
                                             std::string pred) const;
};

}  // namespace op_impl
}  // namespace core
}  // namespace mononn_engine
