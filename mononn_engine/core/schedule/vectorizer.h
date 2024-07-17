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

#include "mononn_engine/core/op/cluster_op.h"
#include "mononn_engine/core/schedule/schedule.h"

namespace mononn_engine {
namespace core {
namespace schedule {
class Vectorizer {
 public:
  using ClusterOp = mononn_engine::core::op::ClusterOp;

  static void vectorize(std::shared_ptr<ClusterOp> cluster_op);

 private:
};

}  // namespace schedule
}  // namespace core
}  // namespace mononn_engine