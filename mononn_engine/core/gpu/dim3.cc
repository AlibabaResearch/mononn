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

#include "mononn_engine/core/gpu/dim3.h"

#include "mononn_engine/helpers/string_helpers.h"

namespace mononn_engine {
namespace core {
namespace gpu {
int Dim3::XYZ() const { return this->x * this->y * this->z; }

std::string Dim3::to_string() const {
  return mononn_engine::helpers::string_format("(%d, %d, %d)", this->x, this->y,
                                               this->z);
}

std::unique_ptr<tensorflow::mononn_extra::proto::Dim3> Dim3::ConvertToProto()
    const {
  std::unique_ptr<tensorflow::mononn_extra::proto::Dim3> dim3 =
      std::make_unique<tensorflow::mononn_extra::proto::Dim3>();

  dim3->set_x(this->x);
  dim3->set_y(this->y);
  dim3->set_z(this->z);

  return std::move(dim3);
}

void Dim3::ParseFromProto(const tensorflow::mononn_extra::proto::Dim3* dim3) {
  this->x = dim3->x();
  this->y = dim3->y();
  this->z = dim3->z();
}
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine