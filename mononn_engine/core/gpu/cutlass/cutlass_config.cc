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

#include "mononn_engine/core/gpu/cutlass/cutlass_config.h"

#include "mononn_engine/helpers/string_helpers.h"

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
std::string CutlassConfig::to_string() const {
  return mononn_engine::helpers::string_format(
      "ThreadBlockShape<%d,%d,%d>, WarpShape<%d,%d,%d>, InstShape<%d, %d, %d>, "
      "Arch %s, stages %d",
      this->ThreadBlockShape.m(), this->ThreadBlockShape.n(),
      this->ThreadBlockShape.k(), this->WarpShape.m(), this->WarpShape.n(),
      this->WarpShape.k(), this->InstructionShape.m(),
      this->InstructionShape.n(), this->InstructionShape.k(),
      this->ArchTag.to_string().c_str(), this->stages);
}

std::unique_ptr<tensorflow::mononn_extra::proto::CutlassConfig>
CutlassConfig::ConvertToProto() const {
  std::unique_ptr<tensorflow::mononn_extra::proto::CutlassConfig>
      cutlass_config =
          std::make_unique<tensorflow::mononn_extra::proto::CutlassConfig>();
  cutlass_config->set_allocated_threadblockshape(
      this->ThreadBlockShape.ConvertToProto().release());
  cutlass_config->set_allocated_warpshape(
      this->WarpShape.ConvertToProto().release());
  cutlass_config->set_allocated_instructionshape(
      this->InstructionShape.ConvertToProto().release());
  cutlass_config->set_allocated_operatorclass(
      this->OperatorClass.ConvertToProto().release());
  cutlass_config->set_allocated_archtag(
      this->ArchTag.ConvertToProto().release());
  cutlass_config->set_stages(this->stages);

  return std::move(cutlass_config);
}

void CutlassConfig::ParseFromProto(
    const tensorflow::mononn_extra::proto::CutlassConfig* cutlass_config) {
  this->ThreadBlockShape.ParseFromProto(&cutlass_config->threadblockshape());
  this->WarpShape.ParseFromProto(&cutlass_config->warpshape());
  this->InstructionShape.ParseFromProto(&cutlass_config->instructionshape());
  this->OperatorClass.ParseFromProto(&cutlass_config->operatorclass());
  this->ArchTag.ParseFromProto(&cutlass_config->archtag());
  this->stages = cutlass_config->stages();
}
}  // namespace cutlass
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine