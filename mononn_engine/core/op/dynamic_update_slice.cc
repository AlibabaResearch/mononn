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

#include "mononn_engine/core/op/dynamic_update_slice.h"

#include "mononn_engine/core/op_impl/dynamic_update_slice_impl.h"

namespace mononn_engine {
namespace core {
namespace op {
using DynamicUpdateSliceImpl =
    mononn_engine::core::op_impl::DynamicUpdateSliceImpl;
using OpImplBase = mononn_engine::core::op_impl::OpImplBase;

OpType DynamicUpdateSlice::get_type() const {
  return OpType::dynamic_update_slice;
}

std::vector<std::shared_ptr<OpImplBase>>
DynamicUpdateSlice::generate_candidate_implementation(
    std::shared_ptr<CUDAContext> context, Tier tier) const {
  DynamicUpdateSliceImpl::InputSpec input_spec;

  for (auto const& operand : this->operands) {
    input_spec.operands.push_back(operand->get_output_tensor(0));
  }

  return DynamicUpdateSliceImpl::get_available_implementations(
      context, input_spec, this->get_output_tensor(0));
}
}  // namespace op
}  // namespace core
}  // namespace mononn_engine