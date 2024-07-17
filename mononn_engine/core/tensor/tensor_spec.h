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

#include <vector>

#include "mononn_engine/core/tensor/dtype.h"
#include "mononn_engine/core/tensor/memory_layout.h"
#include "mononn_engine/core/tensor/tensor_shape.h"

namespace mononn_engine {
namespace core {
namespace tensor {
class TensorSpec {
 public:
  TensorSpec() {}
  TensorSpec(Dtype _dtype, TensorShape _tensor_shape,
             MemoryLayout _memory_layout);

  bool valid() const;

  Dtype get_dtype() const;

  TensorShape get_shape() const;
  int get_shape(int index) const;

  MemoryLayout get_layout() const;
  int get_layout(int index) const;

  int element_count() const;
  int rank() const;

  TensorSpec flatten() const;
  TensorSpec concat(const TensorSpec& rhs) const;
  TensorSpec slice_dim(int start, int end) const;
  TensorSpec reduce_dim(int index) const;
  TensorSpec reshape(std::vector<int> to_shape) const;
  bool can_reshape_to(std::vector<int> to_shape) const;

  TensorSpec tensor_permutation(std::vector<int> perm) const;
  TensorSpec memory_permutation(std::vector<int> perm) const;

  std::string to_string() const;

  TensorSpec vectorize(int vec_len) const;

  int64_t size_in_bytes() const;

  TensorShape get_tensor_shape_with_ordered_memory_layout() const;

  bool operator==(const TensorSpec& rhs) const;
  bool operator!=(const TensorSpec& rhs) const;

 private:
  Dtype dtype;
  TensorShape tensor_shape;
  MemoryLayout memory_layout;
};
}  // namespace tensor
}  // namespace core
}  // namespace mononn_engine