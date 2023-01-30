#pragma once
#include "mononn_engine/core/op/cluster_op.h"

namespace mononn_engine {
namespace core {
namespace op {
class ClusterGemmEpilogue : public ClusterOp {
 public:
  ClusterGemmEpilogue(std::string _name,
                      std::vector<std::shared_ptr<Op>> _operands,
                      std::vector<TensorSpec> _output_specs)
      : ClusterOp(_name, _operands, _output_specs) {}

  TensorShape get_loop_shape() const override;
  Schedule construct_schedule(LocalityTier::Tier tier) override;

  void setup_codegen() override;
  std::string generate_cluster_code() const override;
  ClusterType get_cluster_type() const override;

 private:
};
}  // namespace op
}  // namespace core
}  // namespace mononn_engine