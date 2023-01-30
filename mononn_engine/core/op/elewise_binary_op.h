#pragma once
#include <memory>
#include <vector>

#include "mononn_engine/core/op/op.h"
#include "mononn_engine/core/op_impl/op_impl_base.h"
#include "mononn_engine/core/tensor/tensor_spec.h"

namespace mononn_engine {
namespace core {
namespace op {
class ElewiseBinaryOp : public Op {
 public:
  using OpImpl = mononn_engine::core::op_impl::OpImplBase;
  using TensorSpec = mononn_engine::core::tensor::TensorSpec;

  bool is_elewise_binary() const override;

  ElewiseBinaryOp(std::string _name, std::vector<std::shared_ptr<Op>> _operands,
                  std::vector<TensorSpec> _output_specs)
      : Op(_name, _operands, _output_specs) {}

 private:
};
}  // namespace op
}  // namespace core
}  // namespace mononn_engine