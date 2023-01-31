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

#include "mononn_engine/optimization/pass_runner.h"

#include <algorithm>

#include "mononn_engine/config/config.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace optimization {
using Config = mononn_engine::config::Config;

bool PassRunner::run(Graph* graph, std::shared_ptr<CUDAContext> cuda_context) {
  if (std::find(Config::get()->optimization_disabled_pass.begin(),
                Config::get()->optimization_disabled_pass.end(),
                this->pass->name()) !=
      Config::get()->optimization_disabled_pass.end()) {
    LOG(INFO) << "Optimization pass " << this->pass->name() << " disabled.";
    ++this->run_cnt;
    this->succeed = false;
    return this->succeed;
  }

  LOG(INFO) << "Run optimization pass: " << this->pass->name();

  this->succeed = this->pass->run(graph, cuda_context);
  ++this->run_cnt;

  if (!this->succeed)
    LOG(INFO) << "Pass: " << this->pass->name()
              << " do not have identified pattern";

  return this->succeed;
}
}  // namespace optimization
}  // namespace mononn_engine
