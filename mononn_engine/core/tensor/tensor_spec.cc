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

#include "mononn_engine/core/tensor/tensor_spec.h"

#include "mononn_engine/helpers/macros.h"

namespace mononn_engine {
namespace core {
namespace tensor {
TensorSpec::TensorSpec(Dtype _dtype, TensorShape _tensor_shape,
                       MemoryLayout _memory_layout)
    : dtype(_dtype),
      tensor_shape(_tensor_shape),
      memory_layout(_memory_layout) {}

bool TensorSpec::valid() const {
  return this->memory_layout.valid() &&
         this->memory_layout.rank() == this->tensor_shape.rank();
}

Dtype TensorSpec::get_dtype() const { return this->dtype; }

TensorShape TensorSpec::get_shape() const { return this->tensor_shape; }

int TensorSpec::get_shape(int index) const {
  return this->tensor_shape.get_shape(index);
}

MemoryLayout TensorSpec::get_layout() const { return this->memory_layout; }

int TensorSpec::get_layout(int index) const {
  return this->memory_layout.get(index);
}

int TensorSpec::element_count() const {
  return this->get_shape().element_count();
}

int TensorSpec::rank() const {
  EXPECT_TRUE(this->tensor_shape.rank() == this->memory_layout.rank(),
              "Rank does not match.");

  return this->tensor_shape.rank();
}

TensorSpec TensorSpec::flatten() const {
  return TensorSpec(this->get_dtype(), this->get_shape().flatten(),
                    this->get_layout().flatten());
}

TensorSpec TensorSpec::concat(const TensorSpec& rhs) const {
  return TensorSpec(this->get_dtype(),
                    this->get_shape().concat(rhs.get_shape()),
                    this->get_layout().concat(rhs.get_layout()));
}

TensorSpec TensorSpec::slice_dim(int start, int end) const {
  return TensorSpec(this->get_dtype(), this->get_shape().slice_dim(start, end),
                    this->get_layout().slice_dim(start, end));
}

TensorSpec TensorSpec::reduce_dim(int index) const {
  return TensorSpec(this->get_dtype(), this->get_shape().reduce_dim(index),
                    this->get_layout().reduce_dim(index));
}

TensorSpec TensorSpec::reshape(std::vector<int> to_shape) const {
  if (!this->can_reshape_to(to_shape))
    LOG(FATAL) << "Cannot reshape " << this->to_string() << " to "
               << TensorShape(to_shape).to_string();

  TensorShape new_shape = TensorShape(to_shape);
  std::vector<int> new_layout;

  std::vector<int> curr_shape = this->tensor_shape.get_shape();
  int idx1 = 0;
  int idx2 = 0;
  int acc1, acc2;

  while (idx1 < (int)curr_shape.size() && idx2 < (int)to_shape.size()) {
    acc1 = curr_shape[idx1];
    acc2 = to_shape[idx2];

    if (acc1 == acc2) {
      new_layout.push_back(this->memory_layout.get_layout(idx1) * 1000);

      ++idx1;
      ++idx2;

      continue;
    }

    if (acc1 < acc2) {
      while (acc1 < acc2) {
        if (idx1 + 1 >= (int)curr_shape.size()) LOG(FATAL) << "Failed";
        if (this->memory_layout.get_layout(idx1 + 1) !=
            this->memory_layout.get_layout(idx1) - 1)
          LOG(FATAL) << "Failed";

        acc1 *= curr_shape[++idx1];
      }

      if (acc1 != acc2) LOG(FATAL) << "Failed";

      new_layout.push_back(this->memory_layout.get_layout(idx1) * 1000);

      ++idx1;
      ++idx2;
      continue;
    }

    if (acc1 > acc2) {
      int cnt = 999;
      new_layout.push_back(this->memory_layout.get_layout(idx1) * 1000 + cnt);
      --cnt;

      while (acc1 > acc2) {
        if (idx2 + 1 >= (int)to_shape.size()) LOG(FATAL) << "Failed";

        acc2 *= to_shape[++idx2];

        new_layout.push_back(this->memory_layout.get_layout(idx1) * 1000 + cnt);
        --cnt;
      }

      if (acc1 != acc2) LOG(FATAL) << "Failed";

      ++idx1;
      ++idx2;

      continue;
    }
  }

  if (idx1 != (int)curr_shape.size() || idx2 != to_shape.size())
    LOG(FATAL) << "Failed";

  if (new_shape.get_shape().size() != new_layout.size())
    LOG(FATAL) << "Failed on transformation, new shape size "
               << new_shape.get_shape().size() << " new layout size "
               << new_layout.size();

  return TensorSpec(this->get_dtype(), new_shape,
                    MemoryLayout(new_layout).normalize());
}

bool TensorSpec::can_reshape_to(std::vector<int> to_shape) const {
  std::vector<int> curr_shape = this->tensor_shape.get_shape();
  int idx1 = 0;
  int idx2 = 0;
  int acc1, acc2;

  while (idx1 < (int)curr_shape.size() && idx2 < (int)to_shape.size()) {
    acc1 = curr_shape[idx1];
    acc2 = to_shape[idx2];

    if (acc1 == acc2) {
      ++idx1;
      ++idx2;
      continue;
    }

    if (acc1 < acc2) {
      while (acc1 < acc2) {
        if (idx1 + 1 >= (int)curr_shape.size()) return false;
        if (this->memory_layout.get_layout(idx1 + 1) !=
            this->memory_layout.get_layout(idx1) - 1)
          return false;

        acc1 *= curr_shape[++idx1];
      }

      if (acc1 != acc2) return false;

      ++idx1;
      ++idx2;
      continue;
    }

    if (acc1 > acc2) {
      while (acc1 > acc2) {
        if (idx2 + 1 >= (int)to_shape.size()) return false;

        acc2 *= to_shape[++idx2];
      }

      if (acc1 != acc2) return false;

      ++idx1;
      ++idx2;

      continue;
    }
  }

  return idx1 == (int)curr_shape.size() && idx2 == to_shape.size();
}

TensorSpec TensorSpec::tensor_permutation(std::vector<int> perm) const {
  TensorShape new_shape = this->tensor_shape.permute(perm);
  MemoryLayout new_layout = this->memory_layout.permute(perm);

  return TensorSpec(this->dtype, new_shape, new_layout);
}

TensorSpec TensorSpec::memory_permutation(std::vector<int> perm) const {
  MemoryLayout new_layout = this->memory_layout.permute(perm);
  return TensorSpec(this->dtype, this->tensor_shape, new_layout);
}

std::string TensorSpec::to_string() const {
  return this->get_shape().to_string() + this->get_layout().to_string();
}

TensorSpec TensorSpec::vectorize(int vec_len) const {
  EXPECT_TRUE(this->get_shape(-1) % vec_len == 0,
              "Cannot vectorize tensor with shape " +
                  this->get_shape().to_string() + " to len " +
                  std::to_string(vec_len) + " vector");
  Dtype vec_type = dtype.vectorize(vec_len);
  std::vector<int> vec_tensor_shape = this->tensor_shape.get_shape();
  vec_tensor_shape.back() = vec_tensor_shape.back() / vec_len;

  return TensorSpec(vec_type, vec_tensor_shape, this->memory_layout);
}

int64_t TensorSpec::size_in_bytes() const {
  return int64_t(this->element_count()) * int64_t(this->dtype.size_in_bytes());
}

TensorShape TensorSpec::get_tensor_shape_with_ordered_memory_layout() const {
  std::vector<int> new_shape(this->rank());

  for (int idx = 0; idx < this->rank(); ++idx) {
    new_shape[this->rank() - this->memory_layout.get(idx) - 1] =
        this->tensor_shape.get_shape(idx);
  }

  return new_shape;
}

bool TensorSpec::operator==(const TensorSpec& rhs) const {
  return this->get_shape() == rhs.get_shape() &&
         this->get_layout() == rhs.get_layout() &&
         this->get_dtype() == rhs.get_dtype();
}

bool TensorSpec::operator!=(const TensorSpec& rhs) const {
  return !(*this == rhs);
}
}  // namespace tensor
}  // namespace core
}  // namespace mononn_engine