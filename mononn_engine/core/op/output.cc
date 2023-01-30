#include "mononn_engine/core/op/output.h"

#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/op_impl/output_impl.h"

namespace mononn_engine {
namespace core {
namespace op {
using OpImplBase = mononn_engine::core::op_impl::OpImplBase;
using OutputImpl = mononn_engine::core::op_impl::OutputImpl;
using Tensor = mononn_engine::core::tensor::Tensor;

OpType Output::get_type() const { return OpType::output; }

std::vector<std::shared_ptr<OpImplBase>>
Output::generate_candidate_implementation(std::shared_ptr<CUDAContext> context,
                                          Tier tier) const {
  OutputImpl::InputSpec input_spec;
  input_spec.operand = Tensor(this->get_operand(0)->get_name(),
                              this->get_operand(0)->get_output_spec(0));

  return OutputImpl::get_available_implementations(context, input_spec);
}

void Output::set_output_number(int _output_number) {
  this->output_number = _output_number;
}

int Output::get_output_number() const { return this->output_number; }

}  // namespace op
}  // namespace core
}  // namespace mononn_engine