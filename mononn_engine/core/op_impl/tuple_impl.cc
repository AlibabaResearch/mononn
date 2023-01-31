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

#include "mononn_engine/core/op_impl/tuple_impl.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
std::string TupleImpl::generate_with_index_impl() const { LOG(FATAL) << ""; }

std::vector<Tensor> TupleImpl::get_input_tensor() const {
  return std::vector<Tensor>();
}

std::vector<Tensor> TupleImpl::get_output_tensor() const {
  return std::vector<Tensor>();
}

int TupleImpl::get_elements_per_access() const { return 0; }

void TupleImpl::set_instruction_parallel_factor(int _ilp_factor) {
  LOG(FATAL) << "Unimplemented";
}

std::vector<std::shared_ptr<OpImplBase>>
TupleImpl::get_available_implementations(
    std::shared_ptr<mononn_engine::core::context::CUDAContext> cuda_context,
    TupleImpl::InputSpec input_spec, TupleImpl::Tensor output) {
  return std::vector<std::shared_ptr<OpImplBase>>();
}
}  // namespace op_impl
}  // namespace core
}  // namespace mononn_engine