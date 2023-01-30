#pragma once
#include <memory>
#include <string>
#include <vector>

#include "mononn_engine/core/op/elewise_binary_op.h"
#include "mononn_engine/core/op/op.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/tensor/math_op.h"
#include "mononn_engine/core/tensor/tensor_spec.h"

namespace mononn_engine {
namespace core {
namespace op {
class Compare : public ElewiseBinaryOp {
 public:
  using TensorSpec = mononn_engine::core::tensor::TensorSpec;
  using MathOp = mononn_engine::core::tensor::MathOp;
  using Tier = mononn_engine::core::op_annotation::LocalityTier::Tier;

  Compare(std::string _name, std::vector<std::shared_ptr<Op>> _operands,
          std::vector<TensorSpec> _output_specs)
      : ElewiseBinaryOp(_name, _operands, _output_specs) {}
  OpType get_type() const override;
  std::vector<std::shared_ptr<OpImpl>> generate_candidate_implementation(
      std::shared_ptr<CUDAContext> context, Tier tier) const override;

  void set_comparator(MathOp _comparator);
  MathOp get_comparator() const;

 private:
  MathOp comparator;
};
}  // namespace op
}  // namespace core
}  // namespace mononn_engine