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
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/types/optional.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/tensor/dtype.h"
#include "mononn_engine/core/tensor/math_op.h"

namespace mononn_engine {
namespace core {
namespace gpu {
class Functor {
 public:
  using Dtype = mononn_engine::core::tensor::Dtype;
  using OpType = mononn_engine::core::op::OpType;
  using MathOp = mononn_engine::core::tensor::MathOp;

  Functor() {}
  Functor(std::string _name, Dtype _dtype) : name(_name), dtype(_dtype) {}
  Functor(const char* _name, Dtype _dtype)
      : name(std::string(_name)), dtype(_dtype) {}

  static std::map<std::string, Functor>* registry();
  static std::vector<Dtype> supported_types;
  static std::string get_functor_name_for_op_type(
      OpType op_type, absl::optional<MathOp> math_op = absl::nullopt);
  static std::string get_all_functors_definition();

  std::string get_definition() const;
  std::string get_name() const;
  std::string get_raw_name() const;
  std::string get_functor_type() const;
  Dtype get_dtype() const;

 private:
  std::string name;
  Dtype dtype;
};
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine