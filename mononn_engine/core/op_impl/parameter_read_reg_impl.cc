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

#include "mononn_engine/core/op_impl/parameter_read_reg_impl.h"

#include "mononn_engine/core/gpu/memory.h"
#include "mononn_engine/helpers/helpers.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
using Memory = mononn_engine::core::gpu::Memory;
using Tensor = ParameterReadRegImpl::Tensor;
using TensorSpec = mononn_engine::core::tensor::TensorSpec;

std::string ParameterReadRegImpl::generate_impl() const {
  auto type = this->output.get_dtype();
  std::string node_name = this->output.get_name();

  if (this->is_instruction_parallelized()) {
    std::stringstream ss;

    for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor();
         ++ilp_id) {
      std::string index = mononn_engine::helpers::string_format(
          "%s + %d", this->input_spec.step_id.c_str(), ilp_id);
      ss << Memory::read(
          Memory::AccessFlavor::STRONG_TYPED, type,
          mononn_engine::helpers::get_node_ilp_name(node_name, ilp_id),
          this->input_spec.operand_reg_buffer_name, index, true);
    }

    return ss.str();
  } else {
    return Memory::read(Memory::AccessFlavor::STRONG_TYPED, type, node_name,
                        this->input_spec.operand_reg_buffer_name,
                        this->input_spec.step_id, true);
  }
}

std::vector<Tensor> ParameterReadRegImpl::get_input_tensor() const {
  return {};
}

std::vector<Tensor> ParameterReadRegImpl::get_output_tensor() const {
  return {this->output};
}

int ParameterReadRegImpl::get_elements_per_access() const {
  return this->output.get_dtype().get_elements_per_access();
}

void ParameterReadRegImpl::set_instruction_parallel_factor(int _ilp_factor) {
  this->ilp_factor = _ilp_factor;

  for (auto& [tag, auxiliary_impl] : this->auxiliary_impls) {
    auxiliary_impl->set_instruction_parallel_factor(_ilp_factor);
  }
}

std::vector<std::shared_ptr<OpImplBase>>
ParameterReadRegImpl::get_available_implementations(
    std::shared_ptr<CUDAContext> cuda_context, InputSpec input_spec,
    Tensor output) {
  std::shared_ptr<ParameterReadRegImpl> impl =
      std::make_shared<ParameterReadRegImpl>(cuda_context, input_spec, output);
  return {std::static_pointer_cast<OpImplBase>(impl)};
}
}  // namespace op_impl
}  // namespace core
}  // namespace mononn_engine