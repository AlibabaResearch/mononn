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

#include "mononn_engine/core/common/proto_converter.h"
#include "tensorflow/mononn_extra/proto/dim3.pb.h"

namespace mononn_engine {
namespace core {
namespace gpu {

struct Dim3 : public mononn_engine::core::common::ProtoConverter<
                  tensorflow::mononn_extra::proto::Dim3> {
  Dim3() {}
  Dim3(int _x, int _y, int _z) : x(_x), y(_y), z(_z) {}
  int x, y, z;

  int XYZ() const;

  std::string to_string() const;

  std::unique_ptr<tensorflow::mononn_extra::proto::Dim3> ConvertToProto()
      const override;
  void ParseFromProto(
      tensorflow::mononn_extra::proto::Dim3 const* dim3) override;
};
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine