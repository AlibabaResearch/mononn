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

#include "mononn_engine/core/op_impl/broadcast_impl.h"

#include "mononn_engine/core/gpu/buffer_manager.h"
#include "mononn_engine/core/gpu/memory.h"
#include "mononn_engine/core/tensor/dtype.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
using Tensor = mononn_engine::core::tensor::Tensor;
using Memory = mononn_engine::core::gpu::Memory;
using BufferManager = mononn_engine::core::gpu::BufferManager;
using Dtype = mononn_engine::core::tensor::Dtype;
using CUDAContext = mononn_engine::core::context::CUDAContext;
using InputSpec = BroadcastImpl::InputSpec;
using TensorSpec = mononn_engine::core::tensor::TensorSpec;

std::string BroadcastImpl::generate_impl() const {
  std::string operand_name = this->input_spec.operand.get_name();
  std::string output_name = this->output.get_name();
  Dtype type = this->output.get_dtype();

  std::stringstream ss;

  if (this->is_instruction_parallelized()) {
    for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor();
         ++ilp_id) {
      ss << type.to_string() << " "
         << mononn_engine::helpers::get_node_ilp_name(output_name, ilp_id)
         << " = "
         << mononn_engine::helpers::get_node_ilp_name(operand_name, ilp_id)
         << ";"
         << "\n";
    }
  } else {
    ss << type.to_string() << " " << output_name << " = " << operand_name << ";"
       << "\n";
  }

  return ss.str();
}

std::string BroadcastImpl::generate_with_index_impl() const {
  std::string operand_name = this->input_spec.operand.get_name();
  std::string operand_buffer_name =
      BufferManager::get_buffer_name(operand_name);
  std::string output_name = this->output.get_name();
  Dtype type = this->input_spec.operand.get_dtype();

  if (this->is_instruction_parallelized()) {
    std::stringstream ss;
    ss << type.to_string() << " " << operand_name << ";\n";

    for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor();
         ++ilp_id) {
      std::string ilp_output_name =
          mononn_engine::helpers::get_node_ilp_name(output_name, ilp_id);
      std::string ilp_index =
          this->ilp_concrete_index_list[ilp_id][0].index_after_trace;
      ss << Memory::read(Memory::AccessFlavor::REGULAR, type, ilp_output_name,
                         operand_buffer_name, ilp_index, false);
    }

    return ss.str();
  } else {
    std::string index = this->concrete_index_list[0].index_after_trace;
    return Memory::read(Memory::AccessFlavor::REGULAR, type, output_name,
                        operand_buffer_name, index, true);
  }
}

std::vector<Tensor> BroadcastImpl::get_input_tensor() const {
  return {this->input_spec.operand};
}

std::vector<Tensor> BroadcastImpl::get_output_tensor() const {
  return {this->output};
}

int BroadcastImpl::get_elements_per_access() const {
  return this->input_spec.operand.get_dtype().get_elements_per_access();
}

void BroadcastImpl::set_instruction_parallel_factor(int _ilp_factor) {
  this->ilp_factor = _ilp_factor;

  for (auto& [tag, auxiliary_impl] : this->auxiliary_impls) {
    auxiliary_impl->set_instruction_parallel_factor(_ilp_factor);
  }
}

std::vector<std::shared_ptr<OpImplBase>>
BroadcastImpl::get_available_implementations(
    std::shared_ptr<CUDAContext> cuda_context, InputSpec input_spec,
    Tensor output) {
  std::shared_ptr<BroadcastImpl> broadcast_impl =
      std::make_shared<BroadcastImpl>(cuda_context, input_spec, output);

  return {std::static_pointer_cast<OpImplBase>(broadcast_impl)};
}
}  // namespace op_impl
}  // namespace core
}  // namespace mononn_engine