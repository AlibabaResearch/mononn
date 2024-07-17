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
#include <utility>
#include <vector>

namespace mononn_engine {
namespace core {
namespace tensor {
class TensorShape {
 public:
  TensorShape() = default;
  TensorShape(std::vector<int> _shape) : shape(std::move(_shape)) {}

  int rank() const;
  int get_shape(int index) const;
  void set_shape(int index, int _shape);
  const std::vector<int>& get_shape() const;
  int element_count() const;

  TensorShape flatten() const;
  TensorShape concat(const TensorShape& rhs) const;
  TensorShape concat_on_dim(const TensorShape& rhs, int dim) const;
  TensorShape reduce_dim(int index) const;
  TensorShape slice_dim(int start, int end) const;

  TensorShape reshape(std::vector<int> to_shape) const;
  bool can_reshape_to(std::vector<int> to_shape) const;

  TensorShape permute(std::vector<int> perm) const;

  std::string to_string() const;

  bool is_scalar() const;

  bool operator==(const TensorShape& rhs) const;
  bool operator!=(const TensorShape& rhs) const;

 private:
  std::vector<int> shape;
};
}  // namespace tensor
}  // namespace core
}  // namespace mononn_engine