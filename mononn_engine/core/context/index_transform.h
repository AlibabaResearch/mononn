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
#include <functional>
#include <string>
#include <vector>

#include "mononn_engine/core/tensor/tensor_shape.h"

namespace mononn_engine {
namespace core {
namespace context {
class IndexTransform {
 public:
  using TensorShape = mononn_engine::core::tensor::TensorShape;

  static std::vector<std::string> offset_to_multi_index(
      TensorShape tensor_shape, std::string offset);
  static std::string multi_index_to_offset(
      TensorShape tensor_shape, std::vector<std::string> multi_index);

  static std::function<std::vector<std::string>(std::string)>
  offset_to_multi_index_lazy(TensorShape tensor_shape);
  static std::function<std::string(std::vector<std::string>)>
  multi_index_to_offset_lazy(TensorShape tensor_shape);

 private:
};
}  // namespace context
}  // namespace core
}  // namespace mononn_engine
