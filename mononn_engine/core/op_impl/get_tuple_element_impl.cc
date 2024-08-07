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

#include "mononn_engine/core/op_impl/get_tuple_element_impl.h"

#include <sstream>

#include "mononn_engine/core/gpu/buffer_manager.h"
#include "mononn_engine/core/gpu/multi_buffer.h"
#include "mononn_engine/core/tensor/dtype.h"
#include "mononn_engine/core/tensor/tensor_spec.h"
#include "mononn_engine/helpers/macros.h"
#include "mononn_engine/helpers/string_helpers.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
using Dtype = mononn_engine::core::tensor::Dtype;
using Tensor = mononn_engine::core::tensor::Tensor;
using TensorSpec = mononn_engine::core::tensor::TensorSpec;
using MultiBuffer = mononn_engine::core::gpu::MultiBuffer;
using BufferManager = mononn_engine::core::gpu::BufferManager;
using TensorSpec = mononn_engine::core::tensor::TensorSpec;

std::string GetTupleElementImpl::generate_impl() const {
  if (!this->input_spec.operand.is_tuple()) {
    LOG(FATAL) << "Input operand " << this->input_spec.operand.get_name()
               << " is not tuple";
  }

  MultiBuffer multi_buffer(
      BufferManager::get_buffer_name(this->input_spec.operand.get_name()));
  for (int idx = 0; idx < this->input_spec.operand.tuple_size(); ++idx) {
    multi_buffer.add_buffer(
        this->input_spec.operand.get_tensor_spec_for_tuple(idx));
  }

  std::stringstream ss;
  ss << "void *" << this->output.get_name() << " = ";
  ss << multi_buffer.get_pointer_to_buffer(this->input_spec.tuple_index) << ";";

  return ss.str();
}

std::vector<Tensor> GetTupleElementImpl::get_input_tensor() const {
  return {this->input_spec.operand};
}

std::vector<Tensor> GetTupleElementImpl::get_output_tensor() const {
  return {this->output};
}

int GetTupleElementImpl::get_elements_per_access() const {
  return this->output.get_dtype().get_elements_per_access();
}

void GetTupleElementImpl::set_instruction_parallel_factor(int _ilp_factor) {
  LOG(FATAL) << "Unimplemented";
}

std::vector<std::shared_ptr<OpImplBase>>
GetTupleElementImpl::get_available_implementations(
    std::shared_ptr<mononn_engine::core::context::CUDAContext> cuda_context,
    GetTupleElementImpl::InputSpec input_spec,
    GetTupleElementImpl::Tensor output) {
  std::shared_ptr<GetTupleElementImpl> impl =
      std::make_shared<GetTupleElementImpl>(cuda_context, input_spec, output);

  return {impl};
}
}  // namespace op_impl
}  // namespace core
}  // namespace mononn_engine