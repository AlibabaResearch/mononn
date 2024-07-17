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
#include <memory>
#include <string>
#include <vector>

#include "mononn_engine/core/graph/clustered_graph.h"

namespace mononn_engine {
namespace optimization {
// Optimization pass that operate on clustered graph
class ClusteredGraphPass {
 public:
  using ClusteredGraph = mononn_engine::core::graph::ClusteredGraph;
  virtual std::string name() const = 0;
  virtual bool run(std::shared_ptr<ClusteredGraph> graph) = 0;

 private:
};
}  // namespace optimization
}  // namespace mononn_engine