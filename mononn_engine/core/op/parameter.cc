#include "mononn_engine/core/op/parameter.h"

#include "mononn_engine/core/op_impl/parameter_impl.h"
#include "mononn_engine/core/op_impl/parameter_impl_base.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace core {
namespace op {
using OpImpl = mononn_engine::core::op_impl::OpImplBase;
using Tensor = mononn_engine::core::tensor::Tensor;
using ParameterImplBase = mononn_engine::core::op_impl::ParameterImplBase;
using ParameterImpl = mononn_engine::core::op_impl::ParameterImpl;

OpType Parameter::get_type() const { return OpType::parameter; }

std::vector<std::shared_ptr<OpImpl>>
Parameter::generate_candidate_implementation(
    std::shared_ptr<CUDAContext> context, Tier tier) const {
  Tensor output = this->get_output_tensor(0);

  std::vector<std::shared_ptr<OpImpl>> impls =
      ParameterImpl::get_available_implementations(context, output);

  for (auto& impl : impls) {
    impl->set_hlo_text(this->get_hlo_text());
  }

  return impls;
}

void Parameter::set_parameter_number(int _parameter_number) {
  this->parameter_number = _parameter_number;
}

int Parameter::get_parameter_number() const { return this->parameter_number; }
}  // namespace op
}  // namespace core
}  // namespace mononn_engine