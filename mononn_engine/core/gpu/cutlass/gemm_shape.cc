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

#include "mononn_engine/core/gpu/cutlass/gemm_shape.h"

#include "mononn_engine/helpers/string_helpers.h"
namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
std::string GemmShape::to_string() const {
  return mononn_engine::helpers::string_format(
      "cutlass::gemm::GemmShape<%d, %d, %d>", this->M, this->N, this->K);
}

int GemmShape::m() const { return this->M; }

int GemmShape::n() const { return this->N; }

int GemmShape::k() const { return this->K; }

int GemmShape::mn() const { return this->M * this->N; }

int GemmShape::mk() const { return this->M * this->K; }

int GemmShape::nk() const { return this->N * this->K; }

int GemmShape::mnk() const { return this->M * this->N * this->K; }

std::unique_ptr<tensorflow::mononn_extra::proto::GemmShape>
GemmShape::ConvertToProto() const {
  std::unique_ptr<tensorflow::mononn_extra::proto::GemmShape> gemm_shape =
      std::make_unique<tensorflow::mononn_extra::proto::GemmShape>();

  gemm_shape->set_m(this->M);
  gemm_shape->set_n(this->N);
  gemm_shape->set_k(this->K);

  return std::move(gemm_shape);
}

void GemmShape::ParseFromProto(
    const tensorflow::mononn_extra::proto::GemmShape* gemm_shape) {
  this->M = gemm_shape->m();
  this->N = gemm_shape->n();
  this->K = gemm_shape->k();
}
}  // namespace cutlass
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine