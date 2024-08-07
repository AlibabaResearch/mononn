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

#include "mononn_engine/core/tensor/dtype.h"

namespace mononn_engine {
namespace core {
namespace tensor {
class Scalar {
 public:
  Scalar() = default;
  Scalar(const std::string& _name, const Dtype& _dtype)
      : name(_name), dtype({_dtype}) {}
  Scalar(const std::string& _name, const Dtype& _dtype,
         const std::string& _value)
      : name(_name), dtype({_dtype}), value({_value}) {}
  Scalar(const std::string& _name, const std::vector<Dtype>& _dtype,
         const std::vector<std::string>& _value);

  std::string get_definition() const;
  std::string get_definition_with_value() const;
  std::string get_name() const;
  std::string get_value() const;
  const Dtype& get_dtype() const;

  std::string get_type_string() const;

  bool is_tuple() const;
  const std::vector<Dtype>& get_types_in_list() const;
  const std::vector<std::string>& get_values_in_list() const;

  // Scalar vectorize(int element_per_access) const;
 private:
  std::string name;
  std::vector<Dtype> dtype;
  std::vector<std::string> value;
};
}  // namespace tensor
}  // namespace core
}  // namespace mononn_engine