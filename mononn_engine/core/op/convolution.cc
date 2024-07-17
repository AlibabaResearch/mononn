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

#include "mononn_engine/core/op/convolution.h"

#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace core {
namespace op {
using OpImpl = mononn_engine::core::op_impl::OpImplBase;
OpType Convolution::get_type() const { return OpType::convolution; }

std::vector<std::shared_ptr<OpImpl>>
Convolution::generate_candidate_implementation(
    std::shared_ptr<CUDAContext> context, Tier tier) const {
  LOG(FATAL) << "Not implemented";
}
}  // namespace op
}  // namespace core
}  // namespace mononn_engine