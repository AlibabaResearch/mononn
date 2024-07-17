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

#include <vector>

#include "mononn_engine/core/op_annotation/locality_tier.h"
#include "mononn_engine/core/schedule/loop.h"
#include "mononn_engine/core/tensor/tensor_shape.h"

namespace mononn_engine {
namespace core {
namespace schedule {
class Schedule {
 public:
  using TensorShape = mononn_engine::core::tensor::TensorShape;
  using LocalityTier = mononn_engine::core::op_annotation::LocalityTier;
  using Tier = mononn_engine::core::op_annotation::LocalityTier::Tier;
  Schedule() {}

  void add_loop_schedule(Loop loop);
  int num_loop_schedule() const;
  Loop get_inner_loop() const;

  void set_loop_schedule(int index, Loop loop);
  Loop get_loop_schedule(int index) const;
  TensorShape get_loop_shape() const;
  TensorShape get_loop_shape(int loop_id) const;

  void set_locality_tier(Tier _tier);
  Tier get_locality_tier() const;

  Schedule vectorize(int elements_per_access) const;

 private:
  std::vector<Loop> loop_schedule;
  Tier tier;
};
}  // namespace schedule
}  // namespace core
}  // namespace mononn_engine