#pragma once

#include <string>

#include "mononn_engine/core/tensor/dtype.h"

namespace mononn_engine {
namespace core {
namespace gpu {
class Limits {
 public:
  using Dtype = mononn_engine::core::tensor::Dtype;
  static std::string get_max_positive(Dtype type);
  static std::string get_min_negative(Dtype type);
};

}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine
