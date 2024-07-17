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

#include "mononn_engine/core/op_impl/clamp_impl.h"

#include "mononn_engine/core/gpu/functor.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/semantic/function_invocation.h"
#include "mononn_engine/core/tensor/dtype.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
using OpType = mononn_engine::core::op::OpType;
using Functor = mononn_engine::core::gpu::Functor;
using Dtype = mononn_engine::core::tensor::Dtype;
using FunctionInvocation = mononn_engine::core::semantic::FunctionInvocation;
using Tensor = mononn_engine::core::tensor::Tensor;
using TensorSpec = mononn_engine::core::tensor::TensorSpec;

std::string ClampImpl::generate_impl() const {
  std::string functor_name =
      Functor::get_functor_name_for_op_type(OpType::clamp);
  Dtype type = this->output.get_dtype();
  std::string node_name = this->output.get_name();
  Functor clamp = Functor(functor_name, type);

  std::stringstream ss;
  if (this->is_instruction_parallelized()) {
    for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor();
         ++ilp_id) {
      FunctionInvocation invocation(clamp.get_name());
      invocation.add_arg(mononn_engine::helpers::get_node_ilp_name(
          this->input_spec.min_val.get_name(), ilp_id));
      invocation.add_arg(mononn_engine::helpers::get_node_ilp_name(
          this->input_spec.operand.get_name(), ilp_id));
      invocation.add_arg(mononn_engine::helpers::get_node_ilp_name(
          this->input_spec.max_val.get_name(), ilp_id));

      ss << type.to_string() << " "
         << mononn_engine::helpers::get_node_ilp_name(node_name, ilp_id)
         << " = ";
      ss << invocation.to_string() << ";\n";
    }
  } else {
    FunctionInvocation invocation(clamp.get_name());
    invocation.add_arg(this->input_spec.min_val.get_name());
    invocation.add_arg(this->input_spec.operand.get_name());
    invocation.add_arg(this->input_spec.max_val.get_name());

    ss << type.to_string() << " " << node_name << " = ";
    ss << invocation.to_string() << ";\n";
  }

  return ss.str();
}

int ClampImpl::get_elements_per_access() const {
  return this->output.get_dtype().get_elements_per_access();
}

std::vector<Tensor> ClampImpl::get_input_tensor() const {
  return {this->input_spec.min_val, this->input_spec.operand,
          this->input_spec.max_val};
}

std::vector<Tensor> ClampImpl::get_output_tensor() const {
  return {this->output};
}

void ClampImpl::set_instruction_parallel_factor(int _ilp_factor) {
  this->ilp_factor = _ilp_factor;
  for (auto& [tag, auxiliary_impl] : this->auxiliary_impls) {
    auxiliary_impl->set_instruction_parallel_factor(_ilp_factor);
  }
}

}  // namespace op_impl
}  // namespace core
}  // namespace mononn_engine