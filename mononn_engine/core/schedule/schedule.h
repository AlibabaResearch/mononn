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