#pragma once
#include <functional>
#include <string>
#include <vector>

#include "mononn_engine/core/tensor/tensor_shape.h"

namespace mononn_engine {
namespace core {
namespace context {
class IndexTransform {
 public:
  using TensorShape = mononn_engine::core::tensor::TensorShape;

  static std::vector<std::string> offset_to_multi_index(
      TensorShape tensor_shape, std::string offset);
  static std::string multi_index_to_offset(
      TensorShape tensor_shape, std::vector<std::string> multi_index);

  static std::function<std::vector<std::string>(std::string)>
  offset_to_multi_index_lazy(TensorShape tensor_shape);
  static std::function<std::string(std::vector<std::string>)>
  multi_index_to_offset_lazy(TensorShape tensor_shape);

 private:
};
}  // namespace context
}  // namespace core
}  // namespace mononn_engine
