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

#include "mononn_engine/core/gpu/cutlass/conv_backend_config.h"

#include <algorithm>

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
std::unique_ptr<tensorflow::mononn_extra::proto::ConvBackendConfig>
ConvBackendConfig::ConvertToProto() const {
  std::unique_ptr<tensorflow::mononn_extra::proto::ConvBackendConfig>
      conv_backend_config = std::make_unique<
          tensorflow::mononn_extra::proto::ConvBackendConfig>();

  conv_backend_config->set_conv_result_scale(this->conv_result_scale);
  conv_backend_config->set_side_input_scale(this->side_input_scale);
  conv_backend_config->set_tensor_ops_enabled(this->tensor_ops_enabled);
  conv_backend_config->set_activation_mode(this->activation_mode);

  return std::move(conv_backend_config);
}

void ConvBackendConfig::ParseFromProto(
    const tensorflow::mononn_extra::proto::ConvBackendConfig*
        conv_backend_config) {
  this->conv_result_scale = conv_backend_config->conv_result_scale();
  this->side_input_scale = conv_backend_config->side_input_scale();
  this->tensor_ops_enabled = conv_backend_config->tensor_ops_enabled();
  this->activation_mode = conv_backend_config->activation_mode();
}
}  // namespace cutlass
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine