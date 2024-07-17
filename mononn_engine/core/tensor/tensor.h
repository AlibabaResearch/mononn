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
#include <memory>
#include <string>
#include <vector>

#include "mononn_engine/core/tensor/dtype.h"
#include "mononn_engine/core/tensor/tensor_spec.h"

namespace mononn_engine {
namespace core {
namespace tensor {
class Tensor {
 public:
  Tensor() = default;
  Tensor(std::string _name, TensorSpec _tensor_spec)
      : name(_name), tensor_spec(_tensor_spec) {}

  std::string get_name() const;

  bool valid() const;

  Dtype get_dtype() const;

  TensorShape get_shape() const;
  int get_shape(int index) const;

  MemoryLayout get_layout() const;
  int get_layout(int index) const;

  int element_count() const;

  int rank() const;

  Tensor flatten() const;
  Tensor concat(const Tensor& rhs) const;
  Tensor slice_dim(int start, int end) const;
  Tensor reduce_dim(int index) const;

  std::string to_string() const;

  bool is_tuple() const;
  int tuple_size() const;
  TensorSpec get_tensor_spec() const;
  TensorSpec get_tensor_spec_for_tuple(int index) const;
  void add_additional_tensor_spec_for_tuple(TensorSpec _tensor_spec);

  bool operator==(const Tensor& rhs) const;

 private:
  std::string name;
  TensorSpec tensor_spec;
  std::vector<TensorSpec> additional_tensor_spec_for_tuple;
};
}  // namespace tensor
}  // namespace core
}  // namespace mononn_engine