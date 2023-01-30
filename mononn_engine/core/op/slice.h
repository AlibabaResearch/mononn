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
class Slice : public Op {
 public:
  using TensorSpec = mononn_engine::core::tensor::TensorSpec;
  using Tier = mononn_engine::core::op_annotation::LocalityTier::Tier;

  Slice(std::string _name, std::vector<std::shared_ptr<Op>> _operands,
        std::vector<TensorSpec> _output_specs)
      : Op(_name, _operands, _output_specs) {}
  OpType get_type() const override;
  std::vector<std::shared_ptr<OpImpl>> generate_candidate_implementation(
      std::shared_ptr<CUDAContext> context, Tier tier) const override;

  void set_slice_starts(std::vector<int> _slice_starts);
  std::vector<int> get_slice_starts() const;
  int get_slice_start(int index) const;

  void set_slice_limits(std::vector<int> _slice_limits);
  std::vector<int> get_slice_limits() const;
  int get_slice_limit(int index) const;

  void set_slice_strides(std::vector<int> _slice_strides);
  std::vector<int> get_slice_strides() const;
  int get_slice_stride(int index) const;

 protected:
 private:
  std::vector<int> slice_starts;
  std::vector<int> slice_limits;
  std::vector<int> slice_strides;
};
}  // namespace op
}  // namespace core
}  // namespace mononn_engine