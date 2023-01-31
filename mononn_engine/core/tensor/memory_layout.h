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
class MemoryLayout {
 public:
  MemoryLayout() = default;
  MemoryLayout(std::vector<int> _perm) : perm(std::move(_perm)) {}
  bool valid() const;
  int rank() const;

  std::vector<int> get() const;
  int get(int index) const;

  std::vector<int> get_layout() const;
  int get_layout(int index) const;

  MemoryLayout flatten() const;
  MemoryLayout concat(const MemoryLayout& rhs) const;
  MemoryLayout reduce_dim(int index) const;
  MemoryLayout slice_dim(int start, int end) const;

  MemoryLayout permute(std::vector<int> _perm) const;

  std::string to_string() const;

  MemoryLayout normalize() const;

  bool operator==(const MemoryLayout& rhs) const;

 private:
  // Memory layout from first rank to last rank, 0 means the rank with fastest
  // address variation. This is not following XLA HLO convention.
  std::vector<int> perm;

  void assert_layout_valid() const;
};
}  // namespace tensor
}  // namespace core
}  // namespace mononn_engine