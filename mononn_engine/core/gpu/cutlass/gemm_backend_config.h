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
#include <vector>

#include "mononn_engine/core/common/proto_converter.h"
#include "tensorflow/mononn_extra/proto/gemm_backend_config.pb.h"

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
struct GemmBackendConfig
    : public mononn_engine::core::common::ProtoConverter<
          tensorflow::mononn_extra::proto::GemmBackendConfig> {
  float alpha_real;
  float alpha_imag;
  float beta;
  int lhs_contracting_dimensions;
  int rhs_contracting_dimensions;
  int batch_size;
  std::vector<int> lhs_batch_dimensions;
  std::vector<int> rhs_batch_dimensions;
  int lhs_stride;
  int rhs_stride;
  int selected_algorithm;

  std::string to_string() const;

  std::unique_ptr<tensorflow::mononn_extra::proto::GemmBackendConfig>
  ConvertToProto() const override;
  void ParseFromProto(tensorflow::mononn_extra::proto::GemmBackendConfig const*
                          gemm_backend_config) override;
};
}  // namespace cutlass
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine