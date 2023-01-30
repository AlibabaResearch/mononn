#include "mononn_engine/core/op/reshape.h"

#include "mononn_engine/core/op_impl/reshape_impl.h"
#include "mononn_engine/core/tensor/tensor.h"
#include "mononn_engine/core/tensor/tensor_spec.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace core {
namespace op {
using OpImpl = mononn_engine::core::op_impl::OpImplBase;
using Tensor = mononn_engine::core::tensor::Tensor;
using ReshapeImpl = mononn_engine::core::op_impl::ReshapeImpl;

OpType Reshape::get_type() const { return OpType::reshape; }

std::vector<std::shared_ptr<OpImpl>> Reshape::generate_candidate_implementation(
    std::shared_ptr<CUDAContext> context, Tier tier) const {
  ReshapeImpl::InputSpec input_spec;
  input_spec.operand = this->get_operand(0)->get_output_tensor(0);
  Tensor output = this->get_output_tensor(0);

  std::vector<std::shared_ptr<OpImpl>> impls =
      ReshapeImpl::get_available_implementations(context, input_spec, output);

  for (auto& impl : impls) {
    impl->set_hlo_text(this->get_hlo_text());
  }

  return impls;
}
}  // namespace op
}  // namespace core
}  // namespace mononn_engine