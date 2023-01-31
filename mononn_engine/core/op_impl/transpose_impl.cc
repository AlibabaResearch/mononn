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

#include "mononn_engine/core/op_impl/transpose_impl.h"

#include "mononn_engine/core/tensor/dtype.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
using Tensor = mononn_engine::core::tensor::Tensor;
using OpImplBase = mononn_engine::core::op_impl::OpImplBase;
using Dtype = mononn_engine::core::tensor::Dtype;
using TensorSpec = mononn_engine::core::tensor::TensorSpec;

std::string op_impl::TransposeImpl::generate_impl() const {
  std::string operand_name = this->input_spec.operand.get_name();
  std::string node_name = this->output.get_name();
  Dtype type = this->output.get_dtype();

  std::stringstream ss;

  if (this->is_instruction_parallelized()) {
    for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor();
         ++ilp_id) {
      ss << type.to_string() << " "
         << mononn_engine::helpers::get_node_ilp_name(node_name, ilp_id)
         << " = "
         << mononn_engine::helpers::get_node_ilp_name(operand_name, ilp_id)
         << ";\n";
    }
  } else {
    ss << type.to_string() << " " << node_name << " = " << operand_name
       << ";\n";
  }

  return ss.str();
}

std::vector<Tensor> TransposeImpl::get_input_tensor() const {
  return {this->input_spec.operand};
}

std::vector<Tensor> TransposeImpl::get_output_tensor() const {
  return {this->output};
}

int TransposeImpl::get_elements_per_access() const { return 0; }

void TransposeImpl::set_instruction_parallel_factor(int _ilp_factor) {
  this->ilp_factor = _ilp_factor;

  for (auto& [tag, auxiliary_impl] : this->auxiliary_impls) {
    auxiliary_impl->set_instruction_parallel_factor(_ilp_factor);
  }
}

std::vector<std::shared_ptr<OpImplBase>>
TransposeImpl::get_available_implementations(
    std::shared_ptr<CUDAContext> cuda_context,
    TransposeImpl::InputSpec input_spec, Tensor output) {
  std::shared_ptr<TransposeImpl> transpose_impl =
      std::make_shared<TransposeImpl>(cuda_context, input_spec, output);

  return {std::static_pointer_cast<OpImplBase>(transpose_impl)};
}
}  // namespace op_impl
}  // namespace core
}  // namespace mononn_engine