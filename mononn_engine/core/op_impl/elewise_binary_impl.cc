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

#include "mononn_engine/core/op_impl/elewise_binary_impl.h"

#include <sstream>

#include "mononn_engine/core/gpu/buffer_manager.h"
#include "mononn_engine/core/gpu/functor.h"
#include "mononn_engine/core/gpu/memory.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/semantic/function_invocation.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
using Tensor = mononn_engine::core::tensor::Tensor;
using OpType = mononn_engine::core::op::OpType;
using Functor = mononn_engine::core::gpu::Functor;
using FunctionInvocation = mononn_engine::core::semantic::FunctionInvocation;
using Memory = mononn_engine::core::gpu::Memory;
using BufferManager = mononn_engine::core::gpu::BufferManager;
using TensorSpec = mononn_engine::core::tensor::TensorSpec;
using Dtype = mononn_engine::core::tensor::Dtype;

std::string ElewiseBinaryImpl::generate_impl() const {
  std::stringstream ss;

  if (this->is_instruction_parallelized()) {
    for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor();
         ++ilp_id) {
      auto ilp_invocation =
          this->get_invocation().get_ilp_function_invocation(ilp_id);
      ss << this->output.get_dtype().to_string() << " "
         << mononn_engine::helpers::get_node_ilp_name(this->output.get_name(),
                                                      ilp_id)
         << " = ";
      ss << ilp_invocation.to_string();
      ss << ";\n";
    }
  } else {
    ss << this->output.get_dtype().to_string() << " " << this->output.get_name()
       << " = ";
    ss << this->get_invocation().to_string();
    ss << ";\n";
  }

  return ss.str();
}

std::vector<Tensor> ElewiseBinaryImpl::get_input_tensor() const {
  return {this->input_spec.operand1, this->input_spec.operand2};
}

std::vector<Tensor> ElewiseBinaryImpl::get_output_tensor() const {
  return {this->output};
}

int ElewiseBinaryImpl::get_elements_per_access() const {
  return this->elements_per_access;
}

void ElewiseBinaryImpl::set_instruction_parallel_factor(int _ilp_factor) {
  this->ilp_factor = _ilp_factor;

  for (auto& [tag, auxiliary_impl] : this->auxiliary_impls) {
    auxiliary_impl->set_instruction_parallel_factor(_ilp_factor);
  }

  Dtype type = this->input_spec.op_type == OpType::compare
                   ? this->input_spec.operand1.get_dtype()
                   : this->output.get_dtype();
  Functor functor(this->get_invocation_functor().get_raw_name(), type);
  FunctionInvocation invocation = this->get_invocation();
  invocation.set_func_name(functor.get_name());

  this->set_invocation_functor(functor);
  this->set_invocation(invocation);
}

std::vector<std::shared_ptr<OpImplBase>>
ElewiseBinaryImpl::get_available_implementations(
    std::shared_ptr<CUDAContext> cuda_context, InputSpec input_spec,
    Tensor output) {
  OpType op_type = input_spec.op_type;

  std::string functor_name = Functor::get_functor_name_for_op_type(op_type);
  Functor functor(functor_name, output.get_dtype());

  FunctionInvocation invocation(functor.get_name());
  invocation.add_arg(input_spec.operand1.get_name());
  invocation.add_arg(input_spec.operand2.get_name());

  std::shared_ptr<OpImplBase> op_impl = std::static_pointer_cast<OpImplBase>(
      std::make_shared<ElewiseBinaryImpl>(cuda_context, input_spec, output));

  op_impl->set_invocation_functor(functor);
  op_impl->set_invocation(invocation);

  return {op_impl};
}

std::vector<std::shared_ptr<OpImplBase>>
ElewiseBinaryImpl::get_available_implementations_for_compare(
    std::shared_ptr<CUDAContext> cuda_context, InputSpec input_spec,
    Tensor output, MathOp math_op) {
  OpType op_type = input_spec.op_type;
  std::string functor_name =
      Functor::get_functor_name_for_op_type(op_type, math_op);
  Functor functor(functor_name, input_spec.operand1.get_dtype());

  FunctionInvocation invocation(functor.get_name());
  invocation.add_arg(input_spec.operand1.get_name());
  invocation.add_arg(input_spec.operand2.get_name());

  std::shared_ptr<OpImplBase> op_impl = std::static_pointer_cast<OpImplBase>(
      std::make_shared<ElewiseBinaryImpl>(cuda_context, input_spec, output));

  op_impl->set_invocation_functor(functor);
  op_impl->set_invocation(invocation);

  return {op_impl};
}

std::vector<std::shared_ptr<OpImplBase>>
ElewiseBinaryImpl::get_available_implementations_for_select(
    std::shared_ptr<CUDAContext> cuda_context, InputSpec input_spec,
    Tensor output, Tensor pred) {
  OpType op_type = input_spec.op_type;
  std::string functor_name = Functor::get_functor_name_for_op_type(op_type);
  Functor functor(functor_name, output.get_dtype());

  FunctionInvocation invocation(functor.get_name());
  invocation.add_arg(pred.get_name());
  invocation.add_arg(input_spec.operand1.get_name());
  invocation.add_arg(input_spec.operand2.get_name());

  std::shared_ptr<OpImplBase> op_impl = std::static_pointer_cast<OpImplBase>(
      std::make_shared<ElewiseBinaryImpl>(cuda_context, input_spec, output));

  op_impl->set_invocation_functor(functor);
  op_impl->set_invocation(invocation);

  return {op_impl};
}
}  // namespace op_impl
}  // namespace core
}  // namespace mononn_engine