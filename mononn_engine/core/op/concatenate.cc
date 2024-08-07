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

#include "mononn_engine/core/op/concatenate.h"

#include "mononn_engine/core/op_impl/concatenate_impl.h"
#include "mononn_engine/core/tensor/tensor_spec.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace core {
namespace op {
using OpImpl = mononn_engine::core::op_impl::OpImplBase;
using ConcatenateImpl = mononn_engine::core::op_impl::ConcatenateImpl;

OpType Concatenate::get_type() const { return OpType::concatenate; }

std::vector<std::shared_ptr<OpImpl>>
Concatenate::generate_candidate_implementation(
    std::shared_ptr<CUDAContext> context, Tier tier) const {
  ConcatenateImpl::InputSpec input_spec;
  input_spec.dimension = this->dimension;

  for (auto const& operand : this->operands) {
    Tensor operand_tensor(operand->get_name(), operand->get_output_spec(0));
    input_spec.operands.push_back(operand_tensor);
  }

  Tensor output(this->get_name(), this->get_output_spec(0));

  std::vector<std::shared_ptr<OpImpl>> impls =
      ConcatenateImpl::get_available_implementations(context, input_spec,
                                                     output);

  for (auto& impl : impls) {
    impl->set_hlo_text(this->get_hlo_text());
  }

  return impls;
}

void Concatenate::set_dimension(int _dimension) {
  this->dimension = _dimension;
}

int Concatenate::get_dimension() const { return this->dimension; }
}  // namespace op
}  // namespace core
}  // namespace mononn_engine