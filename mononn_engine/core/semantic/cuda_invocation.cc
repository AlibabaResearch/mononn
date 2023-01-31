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

#include "mononn_engine/core/semantic/cuda_invocation.h"

#include "mononn_engine/helpers/string_helpers.h"

namespace mononn_engine {
namespace core {
namespace semantic {
void CUDAInvocation::add_template_arg(std::string template_arg) {
  this->function_invocation.add_template_arg(template_arg);
}

void CUDAInvocation::add_arg(std::string arg) {
  this->function_invocation.add_arg(arg);
}

std::string CUDAInvocation::cuda_config_to_string() const {
  return mononn_engine::helpers::string_format(
      "<<<dim3(%d, %d, %d), dim3(%d, %d, %d), %d, %s>>>", this->grid.x,
      this->grid.y, this->grid.z, this->block.x, this->block.y, this->block.z,
      this->smem_size, this->stream.c_str());
}

std::string CUDAInvocation::to_string() const {
  return mononn_engine::helpers::string_format(
      "%s%s%s%s", this->function_invocation.get_func_name().c_str(),
      this->function_invocation.template_args_to_string().c_str(),
      this->cuda_config_to_string().c_str(),
      this->function_invocation.args_to_string().c_str());
}
}  // namespace semantic
}  // namespace core
}  // namespace mononn_engine