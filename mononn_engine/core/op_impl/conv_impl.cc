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

#include "mononn_engine/core/op_impl/conv_impl.h"

#include "mononn_engine/core/gpu/buffer_manager.h"
#include "mononn_engine/core/gpu/cutlass/cutlass.h"
#include "mononn_engine/core/gpu/cutlass/shared_storage.h"
#include "mononn_engine/core/semantic/using.h"
#include "mononn_engine/helpers/json.hpp"

namespace mononn_engine {
namespace core {
namespace op_impl {
namespace cutlass = mononn_engine::core::gpu::cutlass;
using Using = mononn_engine::core::semantic::Using;
using Tensor = mononn_engine::core::tensor::Tensor;
using CUDAContext = mononn_engine::core::context::CUDAContext;
using Conv2dProblemSize = mononn_engine::core::gpu::cutlass::Conv2dProblemSize;
using ConvArgument = mononn_engine::core::gpu::cutlass::ConvArgument;
using ConvBackendConfig = mononn_engine::core::gpu::cutlass::ConvBackendConfig;
using CutlassConfig = mononn_engine::core::gpu::cutlass::CutlassConfig;
using SharedStorage = mononn_engine::core::gpu::cutlass::SharedStorage;
using TensorShape = mononn_engine::core::tensor::TensorShape;
using BufferManager = mononn_engine::core::gpu::BufferManager;
using Dtype = mononn_engine::core::tensor::Dtype;

std::string ConvImpl::generate_impl() const {
  std::stringstream ss;
  std::string problem_size_name = this->get_problem_size_name();
  std::string epilogue_name = this->get_epilogue_name();
  std::string conv_argument_name = this->get_conv_argument_name();
  std::string conv_parameter_name = this->get_conv_parameter_name();
  std::string kernel_name = this->get_kernel_name();

  Using epilogue(epilogue_name, "cutlass::epilogue::thread::LinearCombination");

  epilogue.add_template_arg(
      this->output.get_dtype().to_cutlass_type().to_string());
  epilogue.add_template_arg(std::to_string(this->alignmentC));
  ss << epilogue.to_string();

  Using conv_kernel(kernel_name, "cutlass::conv::kernel::DefaultConv2dFprop");
  conv_kernel.add_template_arg(
      this->input_spec.A.get_dtype().to_cutlass_type().to_string());
  conv_kernel.add_template_arg(this->LayoutA.to_string());
  conv_kernel.add_template_arg(
      this->input_spec.B.get_dtype().to_cutlass_type().to_string());
  conv_kernel.add_template_arg(this->LayoutB.to_string());
  conv_kernel.add_template_arg(
      this->output.get_dtype().to_cutlass_type().to_string());
  conv_kernel.add_template_arg(this->LayoutD.to_string());
  conv_kernel.add_template_arg(
      this->output.get_dtype().to_cutlass_type().to_string());
  conv_kernel.add_template_arg(this->cutlass_config.OperatorClass.to_string());
  conv_kernel.add_template_arg(this->cutlass_config.ArchTag.to_string());
  conv_kernel.add_template_arg(
      this->cutlass_config.ThreadBlockShape.to_string());
  conv_kernel.add_template_arg(this->cutlass_config.WarpShape.to_string());
  conv_kernel.add_template_arg(
      this->cutlass_config.InstructionShape.to_string());
  conv_kernel.add_template_arg(epilogue.get_name());
  conv_kernel.add_template_arg(
      cutlass::Swizzle::GemmXThreadblockSwizzle.to_string());
  conv_kernel.add_template_arg(std::to_string(this->cutlass_config.stages));
  conv_kernel.add_template_arg(cutlass::Arch::OpMultiplyAdd.to_string());
  conv_kernel.add_template_arg(
      cutlass::IteratorAlgorithm::kOptimized.to_string());
  conv_kernel.add_template_arg(cutlass::StrideSupport::kStrided.to_string());
  conv_kernel.add_template_arg(std::to_string(this->alignmentA));
  conv_kernel.add_template_arg(std::to_string(this->alignmentB));
  conv_kernel.with("Kernel");
  ss << conv_kernel.to_string();

  ss << this->problem_size.define_variable(problem_size_name);
  ConvArgument conv_codegen_argument = this->conv_arguments;
  conv_codegen_argument.ptr_a =
      BufferManager::get_buffer_name(this->conv_arguments.ptr_a);
  conv_codegen_argument.ptr_b =
      BufferManager::get_buffer_name(this->conv_arguments.ptr_b);
  conv_codegen_argument.ptr_d =
      BufferManager::get_buffer_name(this->conv_arguments.ptr_d);

  if (this->has_bias()) {
    conv_codegen_argument.ptr_c =
        BufferManager::get_buffer_name(this->conv_arguments.ptr_c);
  }

  ss << conv_codegen_argument.define_variable(kernel_name, conv_argument_name);
  ss << mononn_engine::helpers::string_format(
      "typename %s::Params %s(%s, nullptr);\n", kernel_name.c_str(),
      conv_parameter_name.c_str(), conv_argument_name.c_str());
  ss << mononn_engine::helpers::string_format(
      "cutlass_wrapper::cutlass_device_gemm_universal<%s>(%s);\n",
      kernel_name.c_str(), conv_parameter_name.c_str());

  return ss.str();
}

std::vector<Tensor> ConvImpl::get_input_tensor() const {
  if (this->has_bias()) {
    return {this->input_spec.A, this->input_spec.B, *this->input_spec.C};
  } else {
    return {this->input_spec.A, this->input_spec.B};
  }
}

std::vector<Tensor> ConvImpl::get_output_tensor() const {
  return {this->output};
}

int ConvImpl::get_elements_per_access() const {
  return this->output.get_dtype().get_elements_per_access();
}

bool ConvImpl::has_bias() const { return this->input_spec.C != nullptr; }

ConvBackendConfig ConvImpl::get_conv_backend_config() const {
  return this->conv_backend_config;
}

void ConvImpl::set_conv_backend_config(
    const ConvBackendConfig& _conv_backend_config) {
  this->conv_backend_config = _conv_backend_config;
}

int ConvImpl::get_smem_usage_in_bytes() const {
  return SharedStorage::get_shared_storage_size(
      this->cutlass_config.ThreadBlockShape, this->cutlass_config.stages,
      this->input_spec.A.get_dtype().size_in_bytes(),
      this->input_spec.B.get_dtype().size_in_bytes());
}

CutlassConfig ConvImpl::get_cutlass_config() const {
  return this->cutlass_config;
}

void ConvImpl::set_instruction_parallel_factor(int _ilp_factor) {
  LOG(FATAL) << "Unimplemented";
}

std::vector<std::shared_ptr<OpImplBase>>
ConvImpl::get_available_implementations(
    std::shared_ptr<CUDAContext> cuda_context, InputSpec input_spec,
    std::string backend_config_str, Tensor output) {
  std::vector<cutlass::TileDescription> available_tile_description =
      cutlass::TileDescription::get_available_tile_description(
          cuda_context->cuda_device_context.get_cutlass_arch_tag(),
          input_spec.A.get_dtype());

  std::vector<cutlass::TileDescription> valid_tile_description;
  ConvBackendConfig conv_backend_config =
      ConvImpl::parse_conv_backend_config(backend_config_str);

  Dtype A_type = input_spec.A.get_dtype().get_primitive_type();
  Dtype B_type = input_spec.B.get_dtype().get_primitive_type();

  int alignment = 128 / A_type.size_in_bits();

  while (input_spec.A.get_tensor_spec()
                 .get_tensor_shape_with_ordered_memory_layout()
                 .get_shape(3) %
             alignment !=
         0) {
    alignment >>= 1;
  }

  for (auto const& desc : available_tile_description) {
    if (cutlass::SharedStorage::get_shared_storage_size(
            desc.get_ThreadblockShape(), desc.get_stages(),
            A_type.size_in_bytes(), B_type.size_in_bytes()) >
        cuda_context->cuda_runtime_context.smem_size) {
      continue;
    }

    if (cuda_context->cuda_runtime_context.block_dim.XYZ() !=
        desc.threads_per_block()) {
      continue;
    }

    if (alignment == 1 && A_type.size_in_bytes() <= 2 &&
        cutlass::Arch::newer_or_equal(desc.get_ArchTag(),
                                      cutlass::Arch::Sm80)) {
      // async copy in Ampere need at least 4 bytes aligned
      continue;
    }

    valid_tile_description.push_back(desc);
  }

  if (valid_tile_description.empty()) {
    std::stringstream ss;
    ss << "No valid Conv implementation\n";
    ss << cuda_context->cuda_runtime_context.to_string() << "\n";
    ss << cuda_context->cuda_device_context.to_string() << "\n";
    ss << "A: " << input_spec.A.to_string()
       << " B: " << input_spec.B.to_string() << " D: " << output.to_string()
       << "\n";
    ss << "Backend config: " << backend_config_str;
    LOG(WARNING) << ss.str();
  }

  std::vector<std::shared_ptr<OpImplBase>> impls;

  for (auto const& desc : valid_tile_description) {
    CutlassConfig cutlass_config;
    cutlass_config.ThreadBlockShape = desc.get_ThreadblockShape();
    cutlass_config.WarpShape = desc.get_WarpShape();
    cutlass_config.InstructionShape = desc.get_InstructionShape();
    cutlass_config.OperatorClass =
        desc.get_op_class();  // only support tensor cores at this moment;
    cutlass_config.ArchTag = desc.get_ArchTag();
    cutlass_config.stages = desc.get_stages();

    std::shared_ptr<ConvImpl> conv_impl = std::make_shared<ConvImpl>(
        cuda_context, input_spec, cutlass_config, conv_backend_config, output);
    impls.push_back(std::static_pointer_cast<OpImplBase>(conv_impl));
  }

  return impls;
}

ConvBackendConfig ConvImpl::parse_conv_backend_config(std::string config_str) {
  ConvBackendConfig conv_backend_config;
  auto config = nlohmann::json::parse(config_str);
  conv_backend_config.conv_result_scale = config["conv_result_scale"];
  conv_backend_config.side_input_scale = config["side_input_scale"];
  // conv_backend_config.tensor_ops_enabled = config["tensor_ops_enabled"];
  conv_backend_config.tensor_ops_enabled = true;
  conv_backend_config.activation_mode = config["activation_mode"];

  if (!conv_backend_config.tensor_ops_enabled) {
    LOG(WARNING) << "Tensor core not enabled";
    LOG(WARNING) << config_str;
  }

  if (conv_backend_config.activation_mode != "0") {
    LOG(FATAL) << "Activation mode: " << conv_backend_config.activation_mode;
  }

  return conv_backend_config;
}

void ConvImpl::setup() {
  // NHWC
  TensorShape tensor_a_ordered_shape =
      this->input_spec.A.get_tensor_spec()
          .get_tensor_shape_with_ordered_memory_layout();
  // KRSC
  TensorShape tensor_b_ordered_shape =
      this->input_spec.B.get_tensor_spec()
          .get_tensor_shape_with_ordered_memory_layout();
  // NPQK
  TensorShape tensor_d_ordered_shape =
      this->output.get_tensor_spec()
          .get_tensor_shape_with_ordered_memory_layout();

  EXPECT_TRUE(tensor_a_ordered_shape.get_shape(3) ==
                  tensor_b_ordered_shape.get_shape(3),
              "Channel dim not match");
  EXPECT_TRUE(
      tensor_b_ordered_shape.get_shape(1) == this->input_spec.filter_size[0],
      "Filter H dim not match");
  EXPECT_TRUE(
      tensor_b_ordered_shape.get_shape(2) == this->input_spec.filter_size[1],
      "Filter W dim not match");
  EXPECT_TRUE(tensor_d_ordered_shape.get_shape(0) ==
                  tensor_a_ordered_shape.get_shape(0),
              "Batch dim not match");
  EXPECT_TRUE(tensor_d_ordered_shape.get_shape(3) ==
                  tensor_b_ordered_shape.get_shape(0),
              "Filter K dim not match");

  this->problem_size.N = std::to_string(tensor_a_ordered_shape.get_shape(0));
  this->problem_size.H = std::to_string(tensor_a_ordered_shape.get_shape(1));
  this->problem_size.W = std::to_string(tensor_a_ordered_shape.get_shape(2));
  this->problem_size.C = std::to_string(tensor_a_ordered_shape.get_shape(3));

  this->problem_size.K = std::to_string(tensor_b_ordered_shape.get_shape(0));
  this->problem_size.R = std::to_string(this->input_spec.filter_size[0]);
  this->problem_size.S = std::to_string(this->input_spec.filter_size[1]);

  this->problem_size.pad_h = std::to_string(this->input_spec.padding_low[0]);
  this->problem_size.pad_w = std::to_string(this->input_spec.padding_low[1]);

  this->problem_size.stride_h =
      std::to_string(this->input_spec.filter_stride[0]);
  this->problem_size.stride_w =
      std::to_string(this->input_spec.filter_stride[1]);

  this->problem_size.dilation_h = "1";
  this->problem_size.dilation_w = "1";

  this->problem_size.P = std::to_string(tensor_d_ordered_shape.get_shape(1));
  this->problem_size.Q = std::to_string(tensor_d_ordered_shape.get_shape(2));

  this->conv_arguments.problem_size = this->get_problem_size_name();
  this->conv_arguments.ptr_a = this->input_spec.A.get_name();
  this->conv_arguments.ptr_b = this->input_spec.B.get_name();

  if (this->has_bias()) {
    this->conv_arguments.ptr_c = this->input_spec.C->get_name();
  } else {
    this->conv_arguments.ptr_c = "nullptr";
  }

  this->conv_arguments.ptr_d = this->output.get_name();
  this->conv_arguments.alpha =
      std::to_string(this->conv_backend_config.conv_result_scale);
  this->conv_arguments.beta =
      std::to_string(this->conv_backend_config.side_input_scale);

  this->LayoutA = cutlass::Layout::TensorNHWC;
  this->LayoutB = cutlass::Layout::TensorNHWC;
  this->LayoutC = cutlass::Layout::TensorNHWC;
  this->LayoutD = cutlass::Layout::TensorNHWC;

  this->alignmentA = 128 / this->input_spec.A.get_dtype().size_in_bits();
  this->alignmentC = 128 / this->output.get_dtype().size_in_bits();

  if (this->cutlass_config.OperatorClass == cutlass::Arch::OpClassSimt) {
    this->alignmentA = 1;
    this->alignmentB = 1;
    this->alignmentC = 1;
  }

  int channel = tensor_a_ordered_shape.get_shape(3);
  while (channel % this->alignmentA != 0) {
    this->alignmentA >>= 1;
  }

  this->alignmentB = this->alignmentA;

  while (std::stoi(this->problem_size.K) % this->alignmentC != 0) {
    this->alignmentC >>= 1;
  }
}

std::string ConvImpl::get_problem_size_name() const {
  std::string node_name = this->output.get_name();
  return mononn_engine::helpers::string_format("%s_conv_problem_size",
                                               node_name.c_str());
}

std::string ConvImpl::get_epilogue_name() const {
  std::string node_name = this->output.get_name();
  return mononn_engine::helpers::string_format(
      "%s_conv_epilogue_linear_combination", node_name.c_str());
}

std::string ConvImpl::get_conv_argument_name() const {
  std::string node_name = this->output.get_name();
  return mononn_engine::helpers::string_format("%s_conv_arg",
                                               node_name.c_str());
}

std::string ConvImpl::get_conv_parameter_name() const {
  std::string node_name = this->output.get_name();
  return mononn_engine::helpers::string_format("%s_conv_params",
                                               node_name.c_str());
}

std::string ConvImpl::get_kernel_name() const {
  std::string node_name = this->output.get_name();
  return mononn_engine::helpers::string_format("%s_conv_kernel",
                                               node_name.c_str());
}
}  // namespace op_impl
}  // namespace core
}  // namespace mononn_engine
