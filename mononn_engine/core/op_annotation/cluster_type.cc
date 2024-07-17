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

#include "mononn_engine/core/op_annotation/cluster_type.h"

namespace mononn_engine {
namespace core {
namespace op_annotation {
ClusterType const ClusterType::None = "None";
ClusterType const ClusterType::Reduce = "Reduce";
ClusterType const ClusterType::Elewise = "Elewise";
ClusterType const ClusterType::GemmEpilogue = "GemmEpilogue";
ClusterType const ClusterType::Gemm = "Gemm";
ClusterType const ClusterType::Conv = "Conv";
ClusterType const ClusterType::ConvEpilogue = "ConvEpilogue";

std::string ClusterType::to_string() const { return this->name; }

bool ClusterType::operator==(ClusterType const& rhs) const {
  return this->name == rhs.name;
}

bool ClusterType::operator!=(ClusterType const& rhs) const {
  return this->name != rhs.name;
}
}  // namespace op_annotation
}  // namespace core
}  // namespace mononn_engine