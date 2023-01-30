#pragma once

#include <unordered_map>

#include "mononn_engine/core/op_annotation/locality_tier.h"
#include "mononn_engine/core/schedule/schedule.h"
#include "mononn_engine/core/tensor/tensor_shape.h"

namespace mononn_engine {
namespace core {
namespace schedule {
class ScheduleFactory {
 public:
  using LocalityTier = mononn_engine::core::op_annotation::LocalityTier;
  using TensorShape = mononn_engine::core::tensor::TensorShape;
  static Schedule get_schedule(LocalityTier::Tier tier, TensorShape shape);

 private:
};
}  // namespace schedule
}  // namespace core
}  // namespace mononn_engine