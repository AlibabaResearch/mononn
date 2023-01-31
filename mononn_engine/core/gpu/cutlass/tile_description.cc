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

#include "mononn_engine/core/gpu/cutlass/tile_description.h"

#include <functional>

#include "mononn_engine/config/config.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
using Config = mononn_engine::config::Config;

GemmShape TileDescription::get_ThreadblockShape() const {
  return this->ThreadblockShape;
}

GemmShape TileDescription::get_WarpShape() const { return this->WarpShape; }

GemmShape TileDescription::get_InstructionShape() const {
  return this->InstructionShape;
}

cutlass::Arch TileDescription::get_ArchTag() const { return this->ArchTag; }

int TileDescription::get_stages() const { return this->stages; }

cutlass::Arch TileDescription::get_op_class() const {
  if (this->is_simt())
    return cutlass::Arch::OpClassSimt;
  else
    return cutlass::Arch::OpClassTensorOp;
}

bool TileDescription::is_simt() const {
  return this->InstructionShape.mnk() == 1;
}

bool TileDescription::is_tensor_op() const {
  return this->InstructionShape.mnk() != 1;
}

std::vector<TileDescription> TileDescription::get_available_tile_description(
    const cutlass::Arch& arch, const Dtype& data_type) {
  if (data_type != Dtype::FLOAT32 && data_type != Dtype::FLOAT16) {
    LOG(FATAL) << "Unsupported data type: " << data_type.to_string();
  }

  if (!Config::get()->gemm_simt_enabled &&
      !Config::get()->gemm_tensor_op_enabled) {
    LOG(FATAL) << "Both simt and tensor op are disabled;";
  }

  std::vector<TileDescription> tile_descs;

  if (Config::get()->gemm_simt_enabled) {
    auto simt_desc =
        TileDescription::get_available_simt_tile_description(arch, data_type);
    tile_descs.insert(tile_descs.end(), simt_desc.begin(), simt_desc.end());
  }

  if (Config::get()->gemm_tensor_op_enabled) {
    auto tensorop_desc =
        TileDescription::get_available_tensorop_tile_description(arch,
                                                                 data_type);
    tile_descs.insert(tile_descs.end(), tensorop_desc.begin(),
                      tensorop_desc.end());
  }

  return tile_descs;
}

std::vector<TileDescription>
TileDescription::get_available_tensorop_tile_description(
    const cutlass::Arch& arch, const Dtype& data_type) {
  std::vector<TileDescription> descriptions;
  std::vector<TileDescription> descriptions_sm80;
  std::vector<TileDescription> descriptions_sm75;
  std::vector<TileDescription> descriptions_sm70;

  std::function<std::vector<TileDescription>(GemmShape, cutlass::Arch, int)>
      get_tf32_tile_description =
          [&](GemmShape instruction_shape, cutlass::Arch arch_tag,
              int stages) -> std::vector<TileDescription> {
    if (Config::get()->fastest_tuning) {
      return {TileDescription(GemmShape(64, 64, 32), GemmShape(32, 32, 32),
                              instruction_shape, arch_tag, stages),
              TileDescription(GemmShape(64, 128, 32), GemmShape(32, 32, 32),
                              instruction_shape, arch_tag, stages)};
    }

    std::vector<TileDescription> descs;

    for (const int block_k : {16, 32}) {
      descs.push_back(TileDescription(
          GemmShape(256, 128, block_k), GemmShape(64, 64, block_k),
          instruction_shape, arch_tag, stages));  // 256
      descs.push_back(TileDescription(
          GemmShape(128, 256, block_k), GemmShape(64, 64, block_k),
          instruction_shape, arch_tag, stages));  // 256
      descs.push_back(TileDescription(
          GemmShape(128, 128, block_k), GemmShape(32, 64, block_k),
          instruction_shape, arch_tag, stages));  // 256
      descs.push_back(TileDescription(
          GemmShape(128, 64, block_k), GemmShape(32, 32, block_k),
          instruction_shape, arch_tag, stages));  // 256
      descs.push_back(TileDescription(
          GemmShape(64, 128, block_k), GemmShape(32, 32, block_k),
          instruction_shape, arch_tag, stages));  // 256
      descs.push_back(TileDescription(GemmShape(256, 64, block_k),
                                      GemmShape(64, 64, block_k),
                                      instruction_shape, arch_tag, stages));
      descs.push_back(TileDescription(GemmShape(64, 256, block_k),
                                      GemmShape(64, 64, block_k),
                                      instruction_shape, arch_tag, stages));
      descs.push_back(TileDescription(GemmShape(128, 128, block_k),
                                      GemmShape(64, 64, block_k),
                                      instruction_shape, arch_tag, stages));
      descs.push_back(TileDescription(GemmShape(128, 64, block_k),
                                      GemmShape(64, 32, block_k),
                                      instruction_shape, arch_tag, stages));
      descs.push_back(TileDescription(GemmShape(128, 64, block_k),
                                      GemmShape(32, 64, block_k),
                                      instruction_shape, arch_tag, stages));
      descs.push_back(TileDescription(GemmShape(64, 128, block_k),
                                      GemmShape(32, 64, block_k),
                                      instruction_shape, arch_tag, stages));
      descs.push_back(TileDescription(GemmShape(64, 128, block_k),
                                      GemmShape(64, 32, block_k),
                                      instruction_shape, arch_tag, stages));
      descs.push_back(TileDescription(GemmShape(64, 64, block_k),
                                      GemmShape(32, 32, block_k),
                                      instruction_shape, arch_tag, stages));

      if (block_k == 16)
        descs.push_back(TileDescription(GemmShape(32, 32, block_k),
                                        GemmShape(16, 16, block_k),
                                        instruction_shape, arch_tag, stages));
    }

    return descs;
  };

  std::function<std::vector<TileDescription>(GemmShape, cutlass::Arch, int)>
      get_fp16_tile_description =
          [&](GemmShape instruction_shape, cutlass::Arch arch_tag,
              int stages) -> std::vector<TileDescription> {
    if (Config::get()->fastest_tuning) {
      return {TileDescription(GemmShape(64, 64, 32), GemmShape(32, 32, 32),
                              instruction_shape, arch_tag, stages),
              TileDescription(GemmShape(64, 128, 32), GemmShape(32, 32, 32),
                              instruction_shape, arch_tag, stages)};
    }

    std::vector<TileDescription> descs;

    for (const int block_k : {32, 64}) {
      descs.push_back(TileDescription(
          GemmShape(256, 128, block_k), GemmShape(64, 64, block_k),
          instruction_shape, arch_tag, stages));  // 256
      descs.push_back(TileDescription(
          GemmShape(128, 256, block_k), GemmShape(64, 64, block_k),
          instruction_shape, arch_tag, stages));  // 256
      descs.push_back(TileDescription(
          GemmShape(128, 128, block_k), GemmShape(32, 64, block_k),
          instruction_shape, arch_tag, stages));  // 256
      descs.push_back(TileDescription(
          GemmShape(128, 64, block_k), GemmShape(32, 32, block_k),
          instruction_shape, arch_tag, stages));  // 256
      descs.push_back(TileDescription(
          GemmShape(64, 128, block_k), GemmShape(32, 32, block_k),
          instruction_shape, arch_tag, stages));  // 256
      descs.push_back(TileDescription(GemmShape(256, 64, block_k),
                                      GemmShape(64, 64, block_k),
                                      instruction_shape, arch_tag, stages));
      descs.push_back(TileDescription(GemmShape(64, 256, block_k),
                                      GemmShape(64, 64, block_k),
                                      instruction_shape, arch_tag, stages));
      descs.push_back(TileDescription(GemmShape(128, 128, block_k),
                                      GemmShape(64, 64, block_k),
                                      instruction_shape, arch_tag, stages));
      descs.push_back(TileDescription(GemmShape(128, 64, block_k),
                                      GemmShape(64, 32, block_k),
                                      instruction_shape, arch_tag, stages));
      descs.push_back(TileDescription(GemmShape(128, 64, block_k),
                                      GemmShape(32, 64, block_k),
                                      instruction_shape, arch_tag, stages));
      descs.push_back(TileDescription(GemmShape(64, 128, block_k),
                                      GemmShape(32, 64, block_k),
                                      instruction_shape, arch_tag, stages));
      descs.push_back(TileDescription(GemmShape(64, 128, block_k),
                                      GemmShape(64, 32, block_k),
                                      instruction_shape, arch_tag, stages));
      descs.push_back(TileDescription(GemmShape(64, 64, block_k),
                                      GemmShape(32, 32, block_k),
                                      instruction_shape, arch_tag, stages));
    }

    return descs;
  };

  for (int stages = 2; stages <= 10; ++stages) {
    if (Config::get()->faster_tuning) {
      if (stages % 2 != 0) continue;
    }

    if (Config::get()->fastest_tuning) {
      if (stages != 2) continue;
    }

    if (data_type == Dtype::FLOAT16) {
      std::vector<TileDescription> sm80_desc = get_fp16_tile_description(
          GemmShape(16, 8, 16), cutlass::Arch::Sm80, stages);
      descriptions_sm80.insert(descriptions_sm80.end(), sm80_desc.begin(),
                               sm80_desc.end());
    }

    if (data_type == Dtype::FLOAT32 && stages > 2) {  // TF32
      std::vector<TileDescription> sm80_desc = get_tf32_tile_description(
          GemmShape(16, 8, 8), cutlass::Arch::Sm80, stages);
      descriptions_sm80.insert(descriptions_sm80.end(), sm80_desc.begin(),
                               sm80_desc.end());
    }
  }

  if (data_type == Dtype::FLOAT16) {
    descriptions_sm75 =
        get_fp16_tile_description(GemmShape(16, 8, 8), cutlass::Arch::Sm75, 2);
    descriptions_sm70 =
        get_fp16_tile_description(GemmShape(8, 8, 4), cutlass::Arch::Sm70, 2);
  }

  if (arch == cutlass::Arch::Sm80 || arch == cutlass::Arch::Sm86) {
    descriptions.insert(descriptions.end(), descriptions_sm80.begin(),
                        descriptions_sm80.end());
  } else if (arch == cutlass::Arch::Sm75) {
    descriptions.insert(descriptions.end(), descriptions_sm75.begin(),
                        descriptions_sm75.end());
  } else if (arch == cutlass::Arch::Sm70) {
    descriptions.insert(descriptions.end(), descriptions_sm70.begin(),
                        descriptions_sm70.end());
  } else {
    LOG(FATAL) << "Unsupported arch " << arch.to_string();
  }

  return descriptions;
}

std::vector<TileDescription>
TileDescription::get_available_simt_tile_description(const cutlass::Arch& arch,
                                                     const Dtype& data_type) {
  std::function<std::vector<TileDescription>(cutlass::Arch, int)>
      get_tile_description = [&](cutlass::Arch arch_tag,
                                 int stages) -> std::vector<TileDescription> {
    if (Config::get()->fastest_tuning) {
      return {TileDescription(GemmShape(32, 64, 8), GemmShape(16, 16, 8),
                              GemmShape(1, 1, 1), arch_tag, stages),
              TileDescription(GemmShape(32, 32, 8), GemmShape(16, 16, 8),
                              GemmShape(1, 1, 1), arch_tag, stages)};
    }

    std::vector<TileDescription> descs;
    descs.push_back(TileDescription(GemmShape(256, 128, 8),
                                    GemmShape(64, 64, 8), GemmShape(1, 1, 1),
                                    arch_tag, stages));  // 256
    descs.push_back(TileDescription(GemmShape(128, 256, 8),
                                    GemmShape(64, 64, 8), GemmShape(1, 1, 1),
                                    arch_tag, stages));  // 256
    descs.push_back(TileDescription(GemmShape(128, 128, 8),
                                    GemmShape(32, 64, 8), GemmShape(1, 1, 1),
                                    arch_tag, stages));  // 256
    descs.push_back(TileDescription(GemmShape(128, 64, 8), GemmShape(32, 32, 8),
                                    GemmShape(1, 1, 1), arch_tag,
                                    stages));  // 256
    descs.push_back(TileDescription(GemmShape(64, 128, 8), GemmShape(32, 32, 8),
                                    GemmShape(1, 1, 1), arch_tag,
                                    stages));  // 256
    descs.push_back(TileDescription(GemmShape(64, 64, 8), GemmShape(32, 16, 8),
                                    GemmShape(1, 1, 1), arch_tag,
                                    stages));  // 256
    descs.push_back(TileDescription(GemmShape(64, 32, 8), GemmShape(16, 16, 8),
                                    GemmShape(1, 1, 1), arch_tag,
                                    stages));  // 256
    descs.push_back(TileDescription(GemmShape(32, 64, 8), GemmShape(16, 16, 8),
                                    GemmShape(1, 1, 1), arch_tag,
                                    stages));  // 256

    descs.push_back(TileDescription(GemmShape(128, 128, 8),
                                    GemmShape(64, 64, 8), GemmShape(1, 1, 1),
                                    arch_tag, stages));
    descs.push_back(TileDescription(GemmShape(128, 64, 8), GemmShape(64, 32, 8),
                                    GemmShape(1, 1, 1), arch_tag, stages));
    descs.push_back(TileDescription(GemmShape(64, 128, 8), GemmShape(32, 64, 8),
                                    GemmShape(1, 1, 1), arch_tag, stages));
    descs.push_back(TileDescription(GemmShape(64, 64, 8), GemmShape(32, 32, 8),
                                    GemmShape(1, 1, 1), arch_tag, stages));
    descs.push_back(TileDescription(GemmShape(64, 32, 8), GemmShape(32, 16, 8),
                                    GemmShape(1, 1, 1), arch_tag, stages));
    descs.push_back(TileDescription(GemmShape(32, 64, 8), GemmShape(16, 32, 8),
                                    GemmShape(1, 1, 1), arch_tag, stages));
    descs.push_back(TileDescription(GemmShape(32, 32, 8), GemmShape(16, 16, 8),
                                    GemmShape(1, 1, 1), arch_tag, stages));

    return descs;
  };

  if (cutlass::Arch::newer_or_equal(arch, cutlass::Arch::Sm80)) {
    return get_tile_description(cutlass::Arch::Sm75, 2);
    // async copy seems not fast on simt cores. thus not including multistage
    // gemm
  } else {
    return get_tile_description(arch, 2);
  }
}

int TileDescription::threads_per_block() const {
  return (this->ThreadblockShape.m() / this->WarpShape.m()) *
         (this->ThreadblockShape.n() / this->WarpShape.n()) * 32;
}
}  // namespace cutlass
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine