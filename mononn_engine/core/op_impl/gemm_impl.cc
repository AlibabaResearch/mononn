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

#include "mononn_engine/core/op_impl/gemm_impl.h"

#include <sstream>

#include "mononn_engine/core/gpu/buffer_manager.h"
#include "mononn_engine/core/gpu/cutlass/shared_storage.h"
#include "mononn_engine/core/gpu/cutlass/swizzle.h"
#include "mononn_engine/core/gpu/cutlass/tile_description.h"
#include "mononn_engine/core/semantic/function_invocation.h"
#include "mononn_engine/core/semantic/using.h"
#include "mononn_engine/core/tensor/dtype.h"
#include "mononn_engine/helpers/json.hpp"
#include "mononn_engine/helpers/macros.h"
#include "mononn_engine/helpers/string_helpers.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
namespace cutlass = mononn_engine::core::gpu::cutlass;
using Tensor = mononn_engine::core::tensor::Tensor;
using CUDAContext = mononn_engine::core::context::CUDAContext;
using InputSpec = GemmImpl::InputSpec;
using Using = mononn_engine::core::semantic::Using;
using Dtype = mononn_engine::core::tensor::Dtype;
using FunctionInvocation = mononn_engine::core::semantic::FunctionInvocation;
using BufferManager = mononn_engine::core::gpu::BufferManager;
using SharedStorage = mononn_engine::core::gpu::cutlass::SharedStorage;
using TensorSpec = mononn_engine::core::tensor::TensorSpec;

std::string GemmImpl::get_prerequisite_definition() {
  return R"(
namespace cutlass_wrapper{

    template<typename GemmKernel>
    __device__ __forceinline__
    void cutlass_device_gemm_universal(typename GemmKernel::Params &params) {
        cutlass::Device_Gemm_Kernel<GemmKernel>(params);
    }
}
)";
}

std::string GemmImpl::generate_impl() const {
  GemmUniversalArgument codegen_gemm_universal_argument =
      this->gemm_universal_arguments;
  codegen_gemm_universal_argument.ptr_A =
      BufferManager::get_buffer_name(this->gemm_universal_arguments.ptr_A);
  codegen_gemm_universal_argument.ptr_B =
      BufferManager::get_buffer_name(this->gemm_universal_arguments.ptr_B);
  codegen_gemm_universal_argument.ptr_D =
      BufferManager::get_buffer_name(this->gemm_universal_arguments.ptr_D);

  if (this->gemm_universal_arguments.ptr_C != "nullptr") {
    codegen_gemm_universal_argument.ptr_C =
        BufferManager::get_buffer_name(this->gemm_universal_arguments.ptr_C);
  }

  std::stringstream ss;

  Using epilogue =
      Using(this->output.get_name() + "_epilogue_linear_combination",
            "cutlass::epilogue::thread::LinearCombination");
  Using gemm_kernel = Using(this->output.get_name() + "_gemm_kernel",
                            "cutlass::gemm::kernel::DefaultGemmUniversal");

  Dtype output_type = this->output.get_dtype().get_primitive_type();
  Dtype A_type = this->input_spec.A.get_dtype().get_primitive_type();
  Dtype B_type = this->input_spec.B.get_dtype().get_primitive_type();

  epilogue.add_template_arg(output_type.to_cutlass_type().to_string());
  epilogue.add_template_arg(std::to_string(this->alignmentC));

  codegen_gemm_universal_argument.alpha = mononn_engine::helpers::string_format(
      "%s(%s)", output_type.to_cutlass_type().to_string().c_str(),
      this->gemm_universal_arguments.alpha.c_str());
  codegen_gemm_universal_argument.beta = mononn_engine::helpers::string_format(
      "%s(%s)", output_type.to_cutlass_type().to_string().c_str(),
      this->gemm_universal_arguments.beta.c_str());

  ss << epilogue.to_string() << "\n";

  gemm_kernel.add_template_arg(A_type.to_cutlass_type().to_string());
  gemm_kernel.add_template_arg(this->LayoutA.to_string());
  gemm_kernel.add_template_arg("cutlass::ComplexTransform::kNone");
  gemm_kernel.add_template_arg(std::to_string(this->alignmentA));
  gemm_kernel.add_template_arg(B_type.to_cutlass_type().to_string());
  gemm_kernel.add_template_arg(this->LayoutB.to_string());
  gemm_kernel.add_template_arg("cutlass::ComplexTransform::kNone");
  gemm_kernel.add_template_arg(std::to_string(this->alignmentB));
  gemm_kernel.add_template_arg(output_type.to_cutlass_type().to_string());
  gemm_kernel.add_template_arg(this->LayoutD.to_string());
  gemm_kernel.add_template_arg(output_type.to_cutlass_type().to_string());
  gemm_kernel.add_template_arg(this->cutlass_config.OperatorClass.to_string());
  gemm_kernel.add_template_arg(this->cutlass_config.ArchTag.to_string());
  gemm_kernel.add_template_arg(
      this->cutlass_config.ThreadBlockShape.to_string());
  gemm_kernel.add_template_arg(this->cutlass_config.WarpShape.to_string());
  gemm_kernel.add_template_arg(
      this->cutlass_config.InstructionShape.to_string());
  gemm_kernel.add_template_arg(epilogue.get_name());
  gemm_kernel.add_template_arg(
      cutlass::Swizzle::GemmXThreadblockSwizzle.to_string());
  gemm_kernel.add_template_arg(std::to_string(this->cutlass_config.stages));
  gemm_kernel.add_template_arg(cutlass::Arch::OpMultiplyAdd.to_string());
  gemm_kernel.with("GemmKernel");

  ss << gemm_kernel.to_string() << "\n";

  std::string gemm_arg_name = this->output.get_name() + "_gemm_arg";
  std::string gemm_params_name = this->output.get_name() + "_gemm_params";

  ss << codegen_gemm_universal_argument.define_variable(gemm_kernel.get_name(),
                                                        gemm_arg_name)
     << "\n";

  ss << "typename " << gemm_kernel.get_name() << "::Params " << gemm_params_name
     << " = "
     << "cutlass::gemm::device::GemmUniversalBase<" << gemm_kernel.get_name()
     << ">"
     << "::Argument2Params(" << gemm_arg_name << ");\n";

  FunctionInvocation gemm_call(
      "cutlass_wrapper::cutlass_device_gemm_universal");
  gemm_call.add_template_arg(gemm_kernel.get_name());
  gemm_call.add_arg(gemm_params_name);
  ss << gemm_call.to_string() << ";";

  return ss.str();
}

std::vector<Tensor> GemmImpl::get_input_tensor() const {
  if (this->has_bias()) {
    return {this->input_spec.A, this->input_spec.B, *this->input_spec.C};
  } else {
    return {this->input_spec.A, this->input_spec.B};
  }
}

std::vector<Tensor> GemmImpl::get_output_tensor() const {
  return {this->output};
}

int GemmImpl::get_elements_per_access() const {
  return this->output.get_dtype().get_elements_per_access();
}

bool GemmImpl::has_bias() const {
  if (this->input_spec.C) {
    return true;
  } else {
    return false;
  }
}

GemmBackendConfig GemmImpl::get_gemm_backend_config() const {
  return this->gemm_backend_config;
}

int GemmImpl::get_smem_usage_in_bytes() const {
  return SharedStorage::get_shared_storage_size(
      this->cutlass_config.ThreadBlockShape, this->cutlass_config.stages,
      this->input_spec.A.get_dtype().size_in_bytes(),
      this->input_spec.B.get_dtype().size_in_bytes());
}

CutlassConfig GemmImpl::get_cutlass_config() const {
  return this->cutlass_config;
}

GemmBackendConfig GemmImpl::parse_gemm_backend_config(std::string config_str) {
  GemmBackendConfig gemm_backend_config;

  auto config = nlohmann::json::parse(config_str);
  gemm_backend_config.alpha_real = config["alpha_real"];
  gemm_backend_config.alpha_imag = config["alpha_imag"];
  gemm_backend_config.beta = config["beta"];

  if (config["dot_dimension_numbers"]["lhs_contracting_dimensions"].size() !=
      1) {
    LOG(FATAL) << "Lhs contracting dimension num is not one: " << config_str;
  }

  gemm_backend_config.lhs_contracting_dimensions = std::stoi(std::string(
      config["dot_dimension_numbers"]["lhs_contracting_dimensions"][0]));

  if (config["dot_dimension_numbers"]["rhs_contracting_dimensions"].size() !=
      1) {
    LOG(FATAL) << "Rhs contracting dimension num is not one: " << config_str;
  }

  gemm_backend_config.rhs_contracting_dimensions = std::stoi(std::string(
      config["dot_dimension_numbers"]["rhs_contracting_dimensions"][0]));
  gemm_backend_config.batch_size = std::stoi(std::string(config["batch_size"]));

  if (gemm_backend_config.batch_size > 1) {
    std::vector<std::string> lhs_batch_dimensions =
        config["dot_dimension_numbers"]["lhs_batch_dimensions"];
    std::vector<std::string> rhs_batch_dimensions =
        config["dot_dimension_numbers"]["rhs_batch_dimensions"];

    EXPECT_TRUE(lhs_batch_dimensions.size() == rhs_batch_dimensions.size(),
                "Batch dim size should be same");

    for (int idx = 0; idx < (int)lhs_batch_dimensions.size(); ++idx) {
      gemm_backend_config.lhs_batch_dimensions.push_back(
          std::stoi(lhs_batch_dimensions[idx]));
      gemm_backend_config.rhs_batch_dimensions.push_back(
          std::stoi(rhs_batch_dimensions[idx]));
    }
  }

  gemm_backend_config.lhs_stride = std::stoi(std::string(config["lhs_stride"]));
  gemm_backend_config.rhs_stride = std::stoi(std::string(config["rhs_stride"]));

  gemm_backend_config.selected_algorithm =
      std::stoi(std::string(config["selected_algorithm"]));

  return gemm_backend_config;
}

void GemmImpl::setup() {
  if (this->gemm_backend_config.batch_size == 1) {
    this->gemm_universal_arguments.mode = GemmUniversalMode::kGemm;
    this->gemm_universal_arguments.batch_count = "1";
  } else {
    this->gemm_universal_arguments.mode = GemmUniversalMode::kBatched;
    this->gemm_universal_arguments.batch_count =
        std::to_string(this->gemm_backend_config.batch_size);
  }

  Tensor& A = this->input_spec.A;
  Tensor& B = this->input_spec.B;
  Tensor& D = this->output;

  EXPECT_TRUE(
      A.get_shape(this->gemm_backend_config.lhs_contracting_dimensions) ==
          B.get_shape(this->gemm_backend_config.rhs_contracting_dimensions),
      "GEMM contracting dimension not match");

  EXPECT_TRUE(A.rank() == B.rank() && B.rank() == D.rank(),
              "Rank should be same");

  int m, n, k;
  int m_dimension = -1, n_dimension = -1;
  int rank = A.rank();

  k = A.get_shape(this->gemm_backend_config.lhs_contracting_dimensions);
  for (int idx = 0; idx < rank; ++idx) {
    if (idx != this->gemm_backend_config.lhs_contracting_dimensions &&
        std::find(this->gemm_backend_config.lhs_batch_dimensions.begin(),
                  this->gemm_backend_config.lhs_batch_dimensions.end(), idx) ==
            this->gemm_backend_config.lhs_batch_dimensions.end()) {
      EXPECT_TRUE(m_dimension == -1, "Multiple m dims");
      m_dimension = idx;
      m = A.get_shape(idx);
    }

    if (idx != this->gemm_backend_config.rhs_contracting_dimensions &&
        std::find(this->gemm_backend_config.rhs_batch_dimensions.begin(),
                  this->gemm_backend_config.rhs_batch_dimensions.end(), idx) ==
            this->gemm_backend_config.rhs_batch_dimensions.end()) {
      EXPECT_TRUE(n_dimension == -1, "Multiple n dims");
      n_dimension = idx;
      n = B.get_shape(idx);
    }
  }

  // Parse ABCD layout
  this->LayoutA =
      m_dimension < this->gemm_backend_config.lhs_contracting_dimensions
          ? cutlass::Layout::RowMajor
          : cutlass::Layout::ColumnMajor;
  this->LayoutB =
      n_dimension > this->gemm_backend_config.rhs_contracting_dimensions
          ? cutlass::Layout::RowMajor
          : cutlass::Layout::ColumnMajor;

  // Rowmajor example: %cublas-gemm.3 = f16[512,384]{1,0} custom-call(
  // f16[512,128]{1,0} %fusion.37, f16[128,384]{1,0} %constant_45),
  // custom_call_target="__cublas$gemm",
  // backend_config="{
  // \"alpha_real\":1,
  // \"alpha_imag\":0,
  // \"beta\":0,
  // \"dot_dimension_numbers\":{
  // \"lhs_contracting_dimensions\":[\"1\"],
  // \"rhs_contracting_dimensions\":[\"0\"],
  // \"lhs_batch_dimensions\":[],
  // \"rhs_batch_dimensions\":[]},
  // \"batch_size\":\"1\",
  // \"lhs_stride\":\"65536\",
  // \"rhs_stride\":\"49152\",\"selected_algorithm\":\"20\"}"

  // Colmajor example: %cublas-batch-gemm.1 = f16[4,2,128,128]{3,2,1,0}
  // custom-call( f16[4,2,64,128]{3,2,1,0} %fusion.36, f16[4,2,128,64]{3,2,1,0}
  // %get-tuple-element.8), custom_call_target="__cublas$gemm", metadata={
  // backend_config="{\"alpha_real\":1,\
        // "alpha_imag\":0,\"beta\":0,\
        // "dot_dimension_numbers\":{
  // \"lhs_contracting_dimensions\":[\"2\"],
  // \"rhs_contracting_dimensions\":[\"3\"],
  // \"lhs_batch_dimensions\":[\"0\",\"1\"],
  // \"rhs_batch_dimensions\":[\"0\",\"1\"]},
  // \"batch_size\":\"8\",\"lhs_stride\":\"8192\",\"rhs_stride\":\"8192\",\"selected_algorithm\":\"100\"}"
  if (this->LayoutA == cutlass::Layout::RowMajor) {
    EXPECT_TRUE(
        this->gemm_backend_config.lhs_contracting_dimensions + 1 == A.rank(),
        "Ill formed lhs contracting dim " +
            std::to_string(
                this->gemm_backend_config.lhs_contracting_dimensions));
  } else {  // column major
    EXPECT_TRUE(
        this->gemm_backend_config.lhs_contracting_dimensions + 2 == A.rank(),
        "Ill formed lhs contracting dim " +
            std::to_string(
                this->gemm_backend_config.lhs_contracting_dimensions));
  }

  if (this->LayoutB == cutlass::Layout::RowMajor) {
    EXPECT_TRUE(this->gemm_backend_config.rhs_contracting_dimensions + 1 ==
                    B.rank() - 1,
                "Ill formed rhs contracting dim " +
                    std::to_string(
                        this->gemm_backend_config.rhs_contracting_dimensions));
  } else {  // column major
    EXPECT_TRUE(
        this->gemm_backend_config.rhs_contracting_dimensions + 1 == B.rank(),
        "Ill formed rhs contracting dim " +
            std::to_string(
                this->gemm_backend_config.rhs_contracting_dimensions));
  }

  if (D.get_layout(-1) == 0 && D.get_layout(-2) == 1) {  // {1, 0}
    this->LayoutD = cutlass::Layout::RowMajor;
  } else if (D.get_layout(-1) == 1 && D.get_layout(-2) == 0) {  // {0, 1}
    this->LayoutD = cutlass::Layout::ColumnMajor;
  } else {
    LOG(FATAL) << "Unsupported GEMM output layout " << D.to_string();
  }

  this->LayoutC = this->LayoutD;

  if (this->LayoutD == cutlass::Layout::RowMajor) {
    // LOG(WARNING) << "Column major output may need additional calculation";
    this->gemm_universal_arguments.problem_size = cutlass::GemmCoord(m, n, k);
    this->gemm_universal_arguments.alpha =
        std::to_string(this->gemm_backend_config.alpha_real);
    this->gemm_universal_arguments.beta =
        std::to_string(this->gemm_backend_config.beta);
    this->gemm_universal_arguments.ptr_A = A.get_name();
    this->gemm_universal_arguments.ptr_B = B.get_name();
    this->gemm_universal_arguments.ptr_D = D.get_name();
    this->gemm_universal_arguments.batch_stride_A = std::to_string(
        A.get_shape(m_dimension) *
        A.get_shape(this->gemm_backend_config.lhs_contracting_dimensions));
    this->gemm_universal_arguments.batch_stride_B = std::to_string(
        B.get_shape(n_dimension) *
        B.get_shape(this->gemm_backend_config.rhs_contracting_dimensions));
    this->gemm_universal_arguments.batch_stride_D =
        std::to_string(D.get_shape(-1) * D.get_shape(-2));
    this->gemm_universal_arguments.stride_a =
        std::to_string(this->LayoutA == cutlass::Layout::RowMajor ? k : m);
    this->gemm_universal_arguments.stride_b =
        std::to_string(this->LayoutB == cutlass::Layout::RowMajor ? n : k);
    this->gemm_universal_arguments.stride_d =
        std::to_string(this->LayoutD == cutlass::Layout::RowMajor ? n : m);

    EXPECT_TRUE(
        this->gemm_universal_arguments.batch_stride_A ==
            std::to_string(this->gemm_backend_config.lhs_stride),
        mononn_engine::helpers::string_format(
            "Validation Failed, parsed %s, calculated %s",
            std::to_string(this->gemm_backend_config.lhs_stride).c_str(),
            this->gemm_universal_arguments.batch_stride_A.c_str()));
    EXPECT_TRUE(
        this->gemm_universal_arguments.batch_stride_B ==
            std::to_string(this->gemm_backend_config.rhs_stride),
        mononn_engine::helpers::string_format(
            "Validation Failed, parsed %s, calculated %s",
            std::to_string(this->gemm_backend_config.rhs_stride).c_str(),
            this->gemm_universal_arguments.batch_stride_B.c_str()));

    if (this->gemm_backend_config.beta != 0) {
      EXPECT_TRUE(this->input_spec.C, "Expect a valid Gemm bias");
      Tensor const& C = *this->input_spec.C;
      this->gemm_universal_arguments.ptr_C = C.get_name();
      this->gemm_universal_arguments.batch_stride_C =
          std::to_string(C.get_shape(-1) * C.get_shape(-2));
      this->gemm_universal_arguments.stride_c =
          std::to_string(this->LayoutC == cutlass::Layout::RowMajor ? n : m);
    } else {
      this->gemm_universal_arguments.ptr_C = "nullptr";
      this->gemm_universal_arguments.batch_stride_C = "0";
      this->gemm_universal_arguments.stride_c = "0";
    }
  } else {  // Transpose GEMM problem
    this->LayoutC = cutlass::Layout::RowMajor;
    this->LayoutD = cutlass::Layout::RowMajor;
    this->LayoutA = this->LayoutA == cutlass::Layout::RowMajor
                        ? cutlass::Layout::ColumnMajor
                        : cutlass::Layout::RowMajor;
    this->LayoutB = this->LayoutB == cutlass::Layout::RowMajor
                        ? cutlass::Layout::ColumnMajor
                        : cutlass::Layout::RowMajor;

    this->gemm_universal_arguments.problem_size = cutlass::GemmCoord(n, m, k);
    this->gemm_universal_arguments.alpha =
        std::to_string(this->gemm_backend_config.alpha_real);
    this->gemm_universal_arguments.beta =
        std::to_string(this->gemm_backend_config.beta);
    this->gemm_universal_arguments.ptr_A = B.get_name();  // Transposed
    this->gemm_universal_arguments.ptr_B = A.get_name();  // Transposed
    this->gemm_universal_arguments.ptr_D = D.get_name();
    this->gemm_universal_arguments.batch_stride_A = std::to_string(
        B.get_shape(n_dimension) *
        B.get_shape(this->gemm_backend_config.rhs_contracting_dimensions));
    this->gemm_universal_arguments.batch_stride_B = std::to_string(
        A.get_shape(m_dimension) *
        A.get_shape(this->gemm_backend_config.lhs_contracting_dimensions));
    this->gemm_universal_arguments.batch_stride_D =
        std::to_string(D.get_shape(-1) * D.get_shape(-2));
    this->gemm_universal_arguments.stride_a =
        std::to_string(this->LayoutA == cutlass::Layout::RowMajor ? k : n);
    this->gemm_universal_arguments.stride_b =
        std::to_string(this->LayoutB == cutlass::Layout::RowMajor ? m : k);
    this->gemm_universal_arguments.stride_d =
        std::to_string(this->LayoutD == cutlass::Layout::RowMajor ? m : n);

    EXPECT_TRUE(
        this->gemm_universal_arguments.batch_stride_A ==
            std::to_string(this->gemm_backend_config.rhs_stride),
        mononn_engine::helpers::string_format(
            "Validation Failed, parsed %s, calculated %s",
            std::to_string(this->gemm_backend_config.lhs_stride).c_str(),
            this->gemm_universal_arguments.batch_stride_A.c_str()));
    EXPECT_TRUE(
        this->gemm_universal_arguments.batch_stride_B ==
            std::to_string(this->gemm_backend_config.lhs_stride),
        mononn_engine::helpers::string_format(
            "Validation Failed, parsed %s, calculated %s",
            std::to_string(this->gemm_backend_config.rhs_stride).c_str(),
            this->gemm_universal_arguments.batch_stride_B.c_str()));

    if (this->gemm_backend_config.beta != 0) {
      EXPECT_TRUE(this->input_spec.C, "Expect a valid Gemm bias");
      Tensor const& C = *this->input_spec.C;
      this->gemm_universal_arguments.ptr_C = C.get_name();
      this->gemm_universal_arguments.batch_stride_C =
          std::to_string(C.get_shape(-1) * C.get_shape(-2));
      this->gemm_universal_arguments.stride_c =
          std::to_string(this->LayoutC == cutlass::Layout::RowMajor ? m : n);
    } else {
      this->gemm_universal_arguments.ptr_C = "nullptr";
      this->gemm_universal_arguments.batch_stride_C = "0";
      this->gemm_universal_arguments.stride_c = "0";
    }
  }

  // set memory access alignment
  this->alignmentA = 128 / this->input_spec.A.get_dtype().size_in_bits();
  this->alignmentB = 128 / this->input_spec.B.get_dtype().size_in_bits();
  this->alignmentC = 128 / this->output.get_dtype().size_in_bits();

  if (this->cutlass_config.OperatorClass == cutlass::Arch::OpClassSimt) {
    this->alignmentA = 1;
    this->alignmentB = 1;
    this->alignmentC = 1;
  }

  while (std::stoi(this->gemm_universal_arguments.stride_a) %
             this->alignmentA !=
         0) {
    this->alignmentA >>= 1;
  }

  while (std::stoi(this->gemm_universal_arguments.stride_b) %
             this->alignmentB !=
         0) {
    this->alignmentB >>= 1;
  }

  while (std::stoi(this->gemm_universal_arguments.stride_c) %
             this->alignmentC !=
         0) {
    this->alignmentC >>= 1;
  }
}

void GemmImpl::set_instruction_parallel_factor(int _ilp_factor) {
  LOG(FATAL) << "Unimplemented";
}

std::vector<std::shared_ptr<OpImplBase>>
GemmImpl::get_available_implementations(
    std::shared_ptr<CUDAContext> cuda_context, InputSpec input_spec,
    std::string backend_config_str, Tensor output) {
  std::vector<cutlass::TileDescription> available_tile_description =
      cutlass::TileDescription::get_available_tile_description(
          cuda_context->cuda_device_context.get_cutlass_arch_tag(),
          input_spec.A.get_dtype());

  std::vector<cutlass::TileDescription> valid_tile_description;
  GemmBackendConfig gemm_backend_config =
      GemmImpl::parse_gemm_backend_config(backend_config_str);

  Dtype A_type = input_spec.A.get_dtype().get_primitive_type();
  Dtype B_type = input_spec.B.get_dtype().get_primitive_type();

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

    valid_tile_description.push_back(desc);
  }

  if (valid_tile_description.empty()) {
    std::stringstream ss;
    ss << "No valid GEMM implementation\n";
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
    cutlass_config.OperatorClass = desc.get_op_class();
    cutlass_config.ArchTag = desc.get_ArchTag();
    cutlass_config.stages = desc.get_stages();

    std::shared_ptr<GemmImpl> gemm_impl = std::make_shared<GemmImpl>(
        cuda_context, input_spec, cutlass_config, gemm_backend_config, output);
    impls.push_back(std::static_pointer_cast<OpImplBase>(gemm_impl));
  }

  return impls;
}
}  // namespace op_impl
}  // namespace core
}  // namespace mononn_engine