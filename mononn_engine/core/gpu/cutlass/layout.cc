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

#include "mononn_engine/core/gpu/cutlass/layout.h"

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
Layout const Layout::RowMajor = "cutlass::layout::RowMajor";
Layout const Layout::ColumnMajor = "cutlass::layout::ColumnMajor";
Layout const Layout::TensorNHWC = "cutlass::layout::TensorNHWC";
Layout const Layout::TensorNCHW = "cutlass::layout::TensorNCHW";

std::string Layout::to_string() const { return this->name; }

bool Layout::operator==(Layout const& rhs) const {
  return this->name == rhs.name;
}

bool Layout::operator!=(Layout const& rhs) const { return !(*this == rhs); }
}  // namespace cutlass
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine