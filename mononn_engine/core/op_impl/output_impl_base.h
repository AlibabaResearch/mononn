#pragma once
#include <memory>
#include <string>
#include <vector>

#include "mononn_engine/core/context/cuda_context.h"
#include "mononn_engine/core/op_impl/op_impl_base.h"
#include "mononn_engine/core/tensor/tensor.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
class OutputImplBase : public OpImplBase {
 public:
  using CUDAContext = mononn_engine::core::context::CUDAContext;
  using Tensor = mononn_engine::core::tensor::Tensor;

  virtual bool need_pre_inner_loop_generation() const;
  virtual std::string generate_pre_inner_loop() const;

 private:
};
}  // namespace op_impl
}  // namespace core
}  // namespace mononn_engine