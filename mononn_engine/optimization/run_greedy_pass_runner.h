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

#include <memory>

#include "mononn_engine/optimization/pass_runner.h"

namespace mononn_engine {
namespace optimization {
class RunGreedyPassRunner : public PassRunner {
 public:
  RunGreedyPassRunner(std::unique_ptr<GraphPass> _pass)
      : PassRunner(std::move(_pass)) {}
  bool can_run() const override;

 private:
};
}  // namespace optimization
}  // namespace mononn_engine
