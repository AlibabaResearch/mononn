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

#include "mononn_engine/core/graph/graph.h"
#include "mononn_engine/optimization/graph_pass.h"

namespace mononn_engine {
namespace optimization {
class PassRunner {
 public:
  using Graph = mononn_engine::core::graph::Graph;
  using CUDAContext = mononn_engine::core::context::CUDAContext;

  PassRunner(std::unique_ptr<GraphPass> _pass) : pass(std::move(_pass)) {}
  virtual bool can_run() const = 0;
  bool run(Graph* graph, std::shared_ptr<CUDAContext> cuda_ontext);

 protected:
  std::unique_ptr<GraphPass> pass;
  int run_cnt = 0;
  bool succeed = true;
};
}  // namespace optimization
}  // namespace mononn_engine
