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

#include "mononn_engine/core/tensor/tensor.h"

namespace mononn_engine {
namespace core {
namespace tensor {
std::string Tensor::get_name() const { return this->name; }

bool Tensor::valid() const { return this->tensor_spec.valid(); }

Dtype Tensor::get_dtype() const { return this->tensor_spec.get_dtype(); }

TensorShape Tensor::get_shape() const { return this->tensor_spec.get_shape(); }

int Tensor::get_shape(int index) const {
  return this->tensor_spec.get_shape(index);
}

MemoryLayout Tensor::get_layout() const {
  return this->tensor_spec.get_layout();
}

int Tensor::get_layout(int index) const {
  return this->tensor_spec.get_layout(index);
}

int Tensor::element_count() const { return this->tensor_spec.element_count(); }

int Tensor::rank() const { return this->tensor_spec.rank(); }

Tensor Tensor::flatten() const {
  return Tensor(this->name, this->tensor_spec.flatten());
}

Tensor Tensor::concat(const Tensor& rhs) const {
  return Tensor(this->name, this->tensor_spec.concat(rhs.tensor_spec));
}

Tensor Tensor::slice_dim(int start, int end) const {
  return Tensor(this->name, this->tensor_spec.slice_dim(start, end));
}

Tensor Tensor::reduce_dim(int index) const {
  return Tensor(this->name, this->tensor_spec.reduce_dim(index));
}

std::string Tensor::to_string() const {
  return this->get_name() + " " + this->get_dtype().to_string() +
         this->tensor_spec.to_string();
}

bool Tensor::operator==(const Tensor& rhs) const {
  return this->name == rhs.name && this->tensor_spec == rhs.tensor_spec;
}

bool Tensor::is_tuple() const {
  return this->additional_tensor_spec_for_tuple.size() != 0;
}

int Tensor::tuple_size() const {
  return 1 + (int)this->additional_tensor_spec_for_tuple.size();
}

TensorSpec Tensor::get_tensor_spec() const { return this->tensor_spec; }

TensorSpec Tensor::get_tensor_spec_for_tuple(int index) const {
  if (index == 0) return this->tensor_spec;
  return this->additional_tensor_spec_for_tuple[index - 1];
}

void Tensor::add_additional_tensor_spec_for_tuple(TensorSpec _tensor_spec) {
  this->additional_tensor_spec_for_tuple.push_back(_tensor_spec);
}
}  // namespace tensor
}  // namespace core
}  // namespace mononn_engine