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

#include "mononn_engine/core/gpu/cutlass/gemm_backend_config.h"

#include "mononn_engine/helpers/string_helpers.h"

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
std::string GemmBackendConfig::to_string() const {
  return mononn_engine::helpers::string_format(
      "alpha_real %.1f, alpha_img %.1f, beta %.1f, lhs_contracting_dimensions "
      "%d, "
      "rhs_contracting_dimensions %d, batch_size %d, lhs_batch_dimensions %s, "
      "rhs_batch_dimensions %s, lhs stride %d, rhs stride %d, "
      "selected_algorithm %d",
      this->alpha_real, this->alpha_imag, this->beta,
      this->lhs_contracting_dimensions, this->rhs_contracting_dimensions,
      this->batch_size,
      mononn_engine::helpers::to_string(this->lhs_batch_dimensions).c_str(),
      mononn_engine::helpers::to_string(this->rhs_batch_dimensions).c_str(),
      this->lhs_stride, this->rhs_stride, this->selected_algorithm);
}

std::unique_ptr<tensorflow::mononn_extra::proto::GemmBackendConfig>
GemmBackendConfig::ConvertToProto() const {
  std::unique_ptr<tensorflow::mononn_extra::proto::GemmBackendConfig>
      gemm_backend_config = std::make_unique<
          tensorflow::mononn_extra::proto::GemmBackendConfig>();

  gemm_backend_config->set_alpha_real(this->alpha_real);
  gemm_backend_config->set_alpha_imag(this->alpha_imag);
  gemm_backend_config->set_beta(this->beta);
  gemm_backend_config->set_lhs_contracting_dimensions(
      this->lhs_contracting_dimensions);
  gemm_backend_config->set_rhs_contracting_dimensions(
      this->rhs_contracting_dimensions);
  gemm_backend_config->set_batch_size(this->batch_size);
  std::for_each(this->lhs_batch_dimensions.begin(), lhs_batch_dimensions.end(),
                [&](const int& val) -> void {
                  gemm_backend_config->add_lhs_batch_dimensions(val);
                });

  std::for_each(this->rhs_batch_dimensions.begin(), rhs_batch_dimensions.end(),
                [&](const int& val) -> void {
                  gemm_backend_config->add_rhs_batch_dimensions(val);
                });

  gemm_backend_config->set_lhs_stride(this->lhs_stride);
  gemm_backend_config->set_rhs_stride(this->rhs_stride);
  gemm_backend_config->set_selected_algorithm(this->selected_algorithm);

  return std::move(gemm_backend_config);
}

void GemmBackendConfig::ParseFromProto(
    const tensorflow::mononn_extra::proto::GemmBackendConfig*
        gemm_backend_config) {
  this->alpha_real = gemm_backend_config->alpha_real();
  this->alpha_imag = gemm_backend_config->alpha_imag();
  this->beta = gemm_backend_config->beta();
  this->lhs_contracting_dimensions =
      gemm_backend_config->lhs_contracting_dimensions();
  this->rhs_contracting_dimensions =
      gemm_backend_config->rhs_contracting_dimensions();
  this->batch_size = gemm_backend_config->batch_size();

  this->lhs_batch_dimensions =
      std::vector<int>(gemm_backend_config->lhs_batch_dimensions().begin(),
                       gemm_backend_config->lhs_batch_dimensions().end());
  this->rhs_batch_dimensions =
      std::vector<int>(gemm_backend_config->rhs_batch_dimensions().begin(),
                       gemm_backend_config->rhs_batch_dimensions().end());
  this->lhs_stride = gemm_backend_config->lhs_stride();
  this->rhs_stride = gemm_backend_config->rhs_stride();
  this->selected_algorithm = gemm_backend_config->selected_algorithm();
}
}  // namespace cutlass
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine