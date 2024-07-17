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

#include "mononn_engine/optimization/memory_intensive_on_chip_fusion_pass.h"

#include <unordered_set>

#include "mononn_engine/core/op_annotation/cluster_type.h"

namespace mononn_engine {
namespace optimization {
using ClusterType = mononn_engine::core::op_annotation::ClusterType;

std::string MemoryIntensiveOnChipFusionPass::name() const {
  return "MemoryIntensiveOnChipFusionPass";
}

bool MemoryIntensiveOnChipFusionPass::run(Graph* graph) {
  std::unordered_set<std::string> visit;

  return true;
}
}  // namespace optimization
}  // namespace mononn_engine