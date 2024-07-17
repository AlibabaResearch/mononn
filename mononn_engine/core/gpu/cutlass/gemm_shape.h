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
#include "tensorflow/mononn_extra/proto/gemm_shape.pb.h"

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
class GemmShape : public mononn_engine::core::common::ProtoConverter<
                      tensorflow::mononn_extra::proto::GemmShape> {
 public:
  GemmShape() {}
  GemmShape(int _m, int _n, int _k) : M(_m), N(_n), K(_k) {}

  std::string to_string() const;

  int m() const;
  int n() const;
  int k() const;
  int mn() const;
  int mk() const;
  int nk() const;
  int mnk() const;

  std::unique_ptr<tensorflow::mononn_extra::proto::GemmShape> ConvertToProto()
      const override;
  void ParseFromProto(
      tensorflow::mononn_extra::proto::GemmShape const* gemm_shape) override;

 private:
  int M, N, K;
};
}  // namespace cutlass
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine