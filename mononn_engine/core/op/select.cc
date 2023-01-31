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

#include "mononn_engine/core/op/select.h"

#include "mononn_engine/core/op_impl/elewise_binary_impl.h"
#include "mononn_engine/core/tensor/tensor.h"
#include "mononn_engine/core/tensor/tensor_spec.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace core {
namespace op {
using OpImpl = mononn_engine::core::op_impl::OpImplBase;
using Tensor = mononn_engine::core::tensor::Tensor;
using ElewiseBinaryImpl = mononn_engine::core::op_impl::ElewiseBinaryImpl;

OpType Select::get_type() const { return OpType::select; }

std::vector<std::shared_ptr<OpImpl>> Select::generate_candidate_implementation(
    std::shared_ptr<CUDAContext> context, Tier tier) const {
  Tensor pred = this->get_operand(0)->get_output_tensor(0);
  Tensor operand1 = this->get_operand(1)->get_output_tensor(0);
  Tensor operand2 = this->get_operand(2)->get_output_tensor(0);
  Tensor output = this->get_output_tensor(0);

  ElewiseBinaryImpl::InputSpec input_spec;
  input_spec.operand1 = operand1;
  input_spec.operand2 = operand2;
  input_spec.op_type = this->get_type();

  std::vector<std::shared_ptr<OpImpl>> impls =
      ElewiseBinaryImpl::get_available_implementations_for_select(
          context, input_spec, output, pred);

  for (auto& impl : impls) {
    impl->set_hlo_text(this->get_hlo_text());
  }

  return impls;
}
}  // namespace op
}  // namespace core
}  // namespace mononn_engine