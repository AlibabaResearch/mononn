#pragma once
#include <memory>
#include <string>
#include <vector>

#include "mononn_engine/core/op/op.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/tensor/tensor_spec.h"

namespace mononn_engine {
namespace core {
namespace op {
class GlobalSync : public Op {
 public:
  using TensorSpec = mononn_engine::core::tensor::TensorSpec;
  using Tier = mononn_engine::core::op_annotation::LocalityTier::Tier;

  GlobalSync(std::string _name)
      : Op(_name, std::vector<std::shared_ptr<Op>>{},
           std::vector<TensorSpec>{}) {}
  OpType get_type() const override;
  std::vector<std::shared_ptr<OpImpl>> generate_candidate_implementation(
      std::shared_ptr<CUDAContext> context, Tier tier) const override;

 private:
};
}  // namespace op
}  // namespace core
}  // namespace mononn_engine