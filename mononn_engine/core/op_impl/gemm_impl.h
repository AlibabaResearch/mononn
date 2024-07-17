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
#include "mononn_engine/core/gpu/cutlass/cutlass.h"
#include "mononn_engine/core/gpu/cutlass/cutlass_config.h"
#include "mononn_engine/core/gpu/cutlass/gemm_argument.h"
#include "mononn_engine/core/gpu/cutlass/gemm_backend_config.h"
#include "mononn_engine/core/op_impl/op_impl_base.h"
#include "mononn_engine/core/tensor/tensor.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
namespace cutlass = mononn_engine::core::gpu::cutlass;
using CutlassConfig = mononn_engine::core::gpu::cutlass::CutlassConfig;
using GemmBackendConfig = mononn_engine::core::gpu::cutlass::GemmBackendConfig;

class GemmImpl : public OpImplBase {
 public:
  using Tensor = mononn_engine::core::tensor::Tensor;
  using CUDAContext = mononn_engine::core::context::CUDAContext;
  using GemmUniversalArgument =
      mononn_engine::core::gpu::cutlass::GemmUniversalArgument;
  using GemmUniversalMode =
      mononn_engine::core::gpu::cutlass::GemmUniversalMode;

  struct InputSpec {
    Tensor A, B;
    std::shared_ptr<Tensor> C;
  };

  GemmImpl(std::shared_ptr<CUDAContext> _cuda_context, InputSpec _input_spec,
           CutlassConfig _cutlass_config,
           GemmBackendConfig _gemm_backend_config, Tensor _output)
      : cuda_context(_cuda_context),
        input_spec(_input_spec),
        cutlass_config(_cutlass_config),
        gemm_backend_config(_gemm_backend_config),
        output(_output) {
    this->setup();
  }

  std::string generate_impl() const override;

  std::vector<Tensor> get_input_tensor() const override;
  std::vector<Tensor> get_output_tensor() const override;
  int get_elements_per_access() const override;
  bool has_bias() const;
  GemmBackendConfig get_gemm_backend_config() const;
  int get_smem_usage_in_bytes() const override;
  CutlassConfig get_cutlass_config() const;
  void set_instruction_parallel_factor(int _ilp_factor) override;

  static std::vector<std::shared_ptr<OpImplBase>> get_available_implementations(
      std::shared_ptr<CUDAContext> cuda_context, InputSpec input_spec,
      std::string backend_config_str, Tensor output);

  static std::string get_prerequisite_definition();

  static GemmBackendConfig parse_gemm_backend_config(std::string config_str);

 private:
  std::shared_ptr<CUDAContext> cuda_context;
  InputSpec input_spec;
  CutlassConfig cutlass_config;
  Tensor output;

  cutlass::Layout LayoutA;
  cutlass::Layout LayoutB;
  cutlass::Layout LayoutC;
  cutlass::Layout LayoutD;

  int alignmentA;
  int alignmentB;
  int alignmentC;

  GemmBackendConfig gemm_backend_config;
  GemmUniversalArgument gemm_universal_arguments;

  void setup();
};
}  // namespace op_impl
}  // namespace core
}  // namespace mononn_engine