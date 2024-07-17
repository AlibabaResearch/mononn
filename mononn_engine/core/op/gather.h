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

#include "mononn_engine/core/op/op.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/tensor/tensor_spec.h"

namespace mononn_engine {
namespace core {
namespace op {
class Gather : public Op {
 public:
  using TensorSpec = mononn_engine::core::tensor::TensorSpec;
  using Tier = mononn_engine::core::op_annotation::LocalityTier::Tier;

  Gather(std::string _name, std::vector<std::shared_ptr<Op>> _operands,
         std::vector<TensorSpec> _output_specs)
      : Op(_name, _operands, _output_specs) {}
  OpType get_type() const override;
  std::vector<std::shared_ptr<OpImpl>> generate_candidate_implementation(
      std::shared_ptr<CUDAContext> context, Tier tier) const override;

  void set_index_vector_dim(int _index_vector_dim);
  int get_index_vector_dim() const;

  void set_offset_dims(std::vector<int> _offset_dims);
  std::vector<int> get_offset_dims() const;

  void set_slice_sizes(std::vector<int> _slice_sizes);
  std::vector<int> get_slice_sizes() const;

  void set_collapsed_slice_dims(std::vector<int> _collapsed_slice_dims);
  std::vector<int> get_collapsed_slice_dims() const;

  void set_start_index_map(std::vector<int> _start_index_map);
  std::vector<int> get_start_index_map() const;

  void set_indices_are_sorted(bool _indices_are_sorted);
  bool get_indices_are_sorted() const;

  void set_unique_indices(bool _unique_indices);
  bool get_unique_indices() const;

 private:
  int index_vector_dim;
  std::vector<int> offset_dims;
  std::vector<int> slice_sizes;
  std::vector<int> collapsed_slice_dims;
  std::vector<int> start_index_map;
  bool indices_are_sorted;
  bool unique_indices;
};
}  // namespace op
}  // namespace core
}  // namespace mononn_engine