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

#include "mononn_engine/core/op_impl/global_sync_impl.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
using Tensor = GlobalSyncImpl::Tensor;
using TensorSpec = mononn_engine::core::tensor::TensorSpec;

std::string GlobalSyncImpl::generate_impl() const {
  return "synchronization::grid_sync();";
}

std::vector<Tensor> GlobalSyncImpl::get_input_tensor() const { return {}; }

std::vector<Tensor> GlobalSyncImpl::get_output_tensor() const { return {}; }

int GlobalSyncImpl::get_elements_per_access() const { return -1; }

void GlobalSyncImpl::set_instruction_parallel_factor(int _ilp_factor) {
  LOG(FATAL) << "Unimplemented";
}

std::vector<std::shared_ptr<OpImplBase>>
GlobalSyncImpl::get_available_implementations(
    std::shared_ptr<CUDAContext> cuda_context) {
  std::shared_ptr<GlobalSyncImpl> impl =
      std::make_shared<GlobalSyncImpl>(cuda_context);

  return {std::static_pointer_cast<OpImplBase>(impl)};
}
}  // namespace op_impl
}  // namespace core
}  // namespace mononn_engine