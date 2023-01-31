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

#include "mononn_engine/core/schedule/schedule.h"

namespace mononn_engine {
namespace core {
namespace schedule {
using Tier = mononn_engine::core::op_annotation::LocalityTier::Tier;

void Schedule::add_loop_schedule(Loop loop) {
  this->loop_schedule.push_back(loop);
}

int Schedule::num_loop_schedule() const {
  return (int)this->loop_schedule.size();
}

Loop Schedule::get_inner_loop() const { return this->loop_schedule.back(); }

void Schedule::set_loop_schedule(int index, Loop loop) {
  this->loop_schedule[index] = loop;
}

Loop Schedule::get_loop_schedule(int index) const {
  if (index < 0) index = this->num_loop_schedule() + index;
  return this->loop_schedule[index];
}

Schedule::TensorShape Schedule::get_loop_shape() const {
  TensorShape loop_shape = this->loop_schedule[0].get_loop_shape();

  for (int idx = 1; idx < this->num_loop_schedule(); ++idx) {
    loop_shape = loop_shape.concat(this->loop_schedule[idx].get_loop_shape());
  }

  return loop_shape;
}

Schedule::TensorShape Schedule::get_loop_shape(int loop_id) const {
  return this->loop_schedule[loop_id].get_loop_shape();
}

void Schedule::set_locality_tier(Tier _tier) { this->tier = _tier; }

Tier Schedule::get_locality_tier() const { return this->tier; }

Schedule Schedule::vectorize(int elements_per_access) const {
  Schedule new_schedule;
  for (int idx = 0; idx < (int)this->loop_schedule.size(); ++idx) {
    if (idx == (int)this->loop_schedule.size() - 1) {
      new_schedule.add_loop_schedule(
          this->loop_schedule[idx].vectorize(elements_per_access));
    } else {
      new_schedule.add_loop_schedule(this->loop_schedule[idx]);
    }
  }

  new_schedule.set_locality_tier(this->tier);

  return new_schedule;
}
}  // namespace schedule
}  // namespace core
}  // namespace mononn_engine