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

#include "mononn_engine/core/op/get_tuple_element.h"

#include "mononn_engine/core/op_impl/get_tuple_element_impl.h"
#include "mononn_engine/core/tensor/tensor.h"
#include "mononn_engine/core/tensor/tensor_spec.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace core {
namespace op {
using OpImpl = mononn_engine::core::op_impl::OpImplBase;
using Tensor = mononn_engine::core::tensor::Tensor;
using TensorSpec = mononn_engine::core::tensor::TensorSpec;
using GetTupleElementImpl = mononn_engine::core::op_impl::GetTupleElementImpl;

OpType GetTupleElement::get_type() const { return OpType::get_tuple_element; }

void GetTupleElement::set_tuple_index(int _tuple_index) {
  this->tuple_index = _tuple_index;
}

int GetTupleElement::get_tuple_index() const { return this->tuple_index; }

std::vector<std::shared_ptr<OpImpl>>
GetTupleElement::generate_candidate_implementation(
    std::shared_ptr<CUDAContext> context, Tier tier) const {
  GetTupleElementImpl::InputSpec input_spec;
  input_spec.operand = Tensor(this->get_operand(0)->get_name(),
                              this->get_operand(0)->get_output_spec(0));
  input_spec.tuple_index = this->get_tuple_index();

  int tuple_len = this->get_operand(0)->get_output_specs().size();
  for (int idx = 1; idx < tuple_len; ++idx) {
    input_spec.operand.add_additional_tensor_spec_for_tuple(
        this->get_operand(0)->get_output_spec(idx));
  }

  Tensor output(this->get_name(), this->get_output_spec(0));

  std::vector<std::shared_ptr<OpImpl>> impls =
      GetTupleElementImpl::get_available_implementations(context, input_spec,
                                                         output);

  for (auto& impl : impls) {
    impl->set_hlo_text(this->get_hlo_text());
  }

  return impls;
}
}  // namespace op
}  // namespace core
}  // namespace mononn_engine