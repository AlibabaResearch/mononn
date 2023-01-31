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

#include "mononn_engine/core/op/clamp.h"

#include "mononn_engine/core/op_impl/clamp_impl.h"
#include "mononn_engine/core/op_impl/elewise_unary_impl.h"
#include "mononn_engine/core/tensor/tensor.h"
#include "mononn_engine/core/tensor/tensor_spec.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace core {
namespace op {
using OpImpl = mononn_engine::core::op_impl::OpImplBase;
using ElewiseUnaryImpl = mononn_engine::core::op_impl::ElewiseUnaryImpl;
using OpImplBase = mononn_engine::core::op_impl::OpImplBase;
using ClampImpl = mononn_engine::core::op_impl::ClampImpl;
using Tensor = mononn_engine::core::tensor::Tensor;

OpType Clamp::get_type() const { return OpType::clamp; }

std::vector<std::shared_ptr<OpImpl>> Clamp::generate_candidate_implementation(
    std::shared_ptr<CUDAContext> context, Tier tier) const {
  ClampImpl::InputSpec input_spec;
  input_spec.min_val = Tensor(this->get_operand(0)->get_name(),
                              this->get_operand(0)->get_output_spec(0));
  input_spec.operand = Tensor(this->get_operand(1)->get_name(),
                              this->get_operand(1)->get_output_spec(0));
  input_spec.max_val = Tensor(this->get_operand(2)->get_name(),
                              this->get_operand(2)->get_output_spec(0));

  Tensor output(this->get_name(), this->get_output_spec(0));

  std::shared_ptr<ClampImpl> impl =
      std::make_shared<ClampImpl>(context, input_spec, output);

  impl->set_hlo_text(this->get_hlo_text());

  return {std::static_pointer_cast<OpImplBase>(impl)};
}
}  // namespace op
}  // namespace core
}  // namespace mononn_engine