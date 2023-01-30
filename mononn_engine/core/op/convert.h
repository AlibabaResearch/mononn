#pragma once
#include <memory>
#include <string>
#include <vector>

#include "mononn_engine/core/op/elewise_unary_op.h"
#include "mononn_engine/core/op/op.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/tensor/tensor_spec.h"

namespace mononn_engine {
namespace core {
namespace op {
class Convert : public ElewiseUnaryOp {
 public:
  using TensorSpec = mononn_engine::core::tensor::TensorSpec;
  using Tier = mononn_engine::core::op_annotation::LocalityTier::Tier;

  Convert(std::string _name, std::vector<std::shared_ptr<Op>> _operands,
          std::vector<TensorSpec> _output_specs)
      : ElewiseUnaryOp(_name, _operands, _output_specs) {}
  OpType get_type() const override;
  std::vector<std::shared_ptr<OpImpl>> generate_candidate_implementation(
      std::shared_ptr<CUDAContext> context, Tier tier) const override;

 private:
};
}  // namespace op
}  // namespace core
}  // namespace mononn_engine