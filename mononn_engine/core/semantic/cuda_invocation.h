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

#pragma once

#include <string>

#include "mononn_engine/core/gpu/dim3.h"
#include "mononn_engine/core/semantic/function_invocation.h"

namespace mononn_engine {
namespace core {
namespace semantic {
class CUDAInvocation {
 public:
  using Dim3 = mononn_engine::core::gpu::Dim3;
  CUDAInvocation(std::string _func_name, Dim3 _grid, Dim3 _block,
                 int _smem_size, std::string _stream)
      : function_invocation(_func_name),
        grid(_grid),
        block(_block),
        smem_size(_smem_size),
        stream(_stream) {}

  void add_template_arg(std::string template_arg);
  void add_arg(std::string arg);

  std::string cuda_config_to_string() const;
  std::string to_string() const;

 private:
  FunctionInvocation function_invocation;
  Dim3 grid;
  Dim3 block;
  int smem_size;
  std::string stream;
};
}  // namespace semantic
}  // namespace core
}  // namespace mononn_engine