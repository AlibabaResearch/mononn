#include "mononn_engine/core/op/copy.h"

#include "mononn_engine/core/op_impl/elewise_unary_impl.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace core {
namespace op {
using OpImpl = mononn_engine::core::op_impl::OpImplBase;
using Tensor = mononn_engine::core::tensor::Tensor;
using ElewiseUnaryImpl = mononn_engine::core::op_impl::ElewiseUnaryImpl;
using OpType = mononn_engine::core::op::OpType;

OpType Copy::get_type() const { return OpType::copy; }

std::vector<std::shared_ptr<OpImpl>> Copy::generate_candidate_implementation(
    std::shared_ptr<CUDAContext> context, Tier tier) const {
  ElewiseUnaryImpl::InputSpec input_spec;
  input_spec.op_type = OpType::convert;
  input_spec.operand = this->get_operand(0)->get_output_tensor(0);
  Tensor output = this->get_output_tensor(0);

  std::vector<std::shared_ptr<OpImpl>> impls =
      ElewiseUnaryImpl::get_available_implementations(context, input_spec,
                                                      output);

  for (auto& impl : impls) {
    impl->set_hlo_text(this->get_hlo_text());
  }

  return impls;
}
}  // namespace op
}  // namespace core
}  // namespace mononn_engine