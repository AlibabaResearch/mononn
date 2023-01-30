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
class Pad : public Op {
 public:
  using TensorSpec = mononn_engine::core::tensor::TensorSpec;
  using Tier = mononn_engine::core::op_annotation::LocalityTier::Tier;

  Pad(std::string _name, std::vector<std::shared_ptr<Op>> _operands,
      std::vector<TensorSpec> _output_specs)
      : Op(_name, _operands, _output_specs) {}
  OpType get_type() const override;
  std::vector<std::shared_ptr<OpImpl>> generate_candidate_implementation(
      std::shared_ptr<CUDAContext> context, Tier tier) const override;

  void set_padding_low(std::vector<int> _padding_low);
  std::vector<int> get_padding_low() const;

  void set_padding_high(std::vector<int> _padding_high);
  std::vector<int> get_padding_high() const;

  //        bool need_post_inner_loop_generation() const override;
  //        std::string generate_post_inner_loop() const override;
 private:
  std::vector<int> padding_low;
  std::vector<int> padding_high;
};
}  // namespace op
}  // namespace core
}  // namespace mononn_engine