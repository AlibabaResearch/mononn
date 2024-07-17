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

#include "mononn_engine/core/gpu/cutlass/arch.h"
#include "mononn_engine/core/gpu/cutlass/gemm_shape.h"
#include "mononn_engine/core/tensor/dtype.h"

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
using Dtype = mononn_engine::core::tensor::Dtype;

class TileDescription {
 public:
  TileDescription(GemmShape _ThreadblockShape, GemmShape _WarpShape,
                  GemmShape _InstructionShape, cutlass::Arch _ArchTag,
                  int _stages)
      : ThreadblockShape(_ThreadblockShape),
        WarpShape(_WarpShape),
        InstructionShape(_InstructionShape),
        ArchTag(_ArchTag),
        stages(_stages) {}
  static std::vector<TileDescription> get_available_tile_description(
      const cutlass::Arch& arch, const Dtype& data_type);
  static std::vector<TileDescription> get_available_tensorop_tile_description(
      const cutlass::Arch& arch, const Dtype& data_type);
  static std::vector<TileDescription> get_available_simt_tile_description(
      const cutlass::Arch& arch, const Dtype& data_type);

  int threads_per_block() const;

  GemmShape get_ThreadblockShape() const;
  GemmShape get_WarpShape() const;
  GemmShape get_InstructionShape() const;
  cutlass::Arch get_ArchTag() const;
  int get_stages() const;
  cutlass::Arch get_op_class() const;
  bool is_simt() const;
  bool is_tensor_op() const;

 private:
  GemmShape ThreadblockShape;
  GemmShape WarpShape;
  GemmShape InstructionShape;
  cutlass::Arch ArchTag;
  int stages;
};
}  // namespace cutlass
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine