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

#include "mononn_engine/core/op/cluster_gemm_epilogue.h"

#include "mononn_engine/core/op_annotation/cluster_type.h"
#include "mononn_engine/core/op_annotation/locality_tier.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace core {
namespace op {
using Schedule = mononn_engine::core::schedule::Schedule;
using LocalityTier = mononn_engine::core::op_annotation::LocalityTier;
using TensorShape = mononn_engine::core::tensor::TensorShape;
using ClusterType = mononn_engine::core::op_annotation::ClusterType;

TensorShape ClusterGemmEpilogue::get_loop_shape() const {
  LOG(FATAL) << "Loop shape undefined";
}

Schedule ClusterGemmEpilogue::construct_schedule(LocalityTier::Tier tier) {
  Schedule schedule;
  schedule.set_locality_tier(LocalityTier::kT3);

  return {schedule};
}

void ClusterGemmEpilogue::setup_codegen() {}

std::string ClusterGemmEpilogue::generate_cluster_code() const {
  LOG(FATAL) << "Not implemented";
}

ClusterType ClusterGemmEpilogue::get_cluster_type() const {
  return ClusterType::GemmEpilogue;
}
}  // namespace op
}  // namespace core
}  // namespace mononn_engine