#include "mononn_engine/core/op_impl/parameter_smem_impl.h"

#include <sstream>

#include "mononn_engine/helpers/string_helpers.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
using Tensor = mononn_engine::core::tensor::Tensor;
using TensorSpec = mononn_engine::core::tensor::TensorSpec;

std::string ParameterSmemImpl::generate_impl() const {
  std::stringstream ss;
  auto type = this->output.get_dtype();

  if (this->is_instruction_parallelized()) {
    for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor();
         ++ilp_id) {
      ss << mononn_engine::helpers::string_format(
                "%s %s = %s;", type.to_string().c_str(),
                mononn_engine::helpers::get_node_ilp_name(
                    this->output.get_name().c_str(), ilp_id)
                    .c_str(),
                this->input_spec.operand.get_name().c_str())
         << "\n";
    }
  } else {
    ss << mononn_engine::helpers::string_format(
              "%s %s = %s;", type.to_string().c_str(),
              this->output.get_name().c_str(),
              this->input_spec.operand.get_name().c_str())
       << "\n";
  }

  return ss.str();
}

std::vector<Tensor> ParameterSmemImpl::get_input_tensor() const {
  return {this->input_spec.operand};
}

std::vector<Tensor> ParameterSmemImpl::get_output_tensor() const {
  return {this->output};
}

int ParameterSmemImpl::get_elements_per_access() const {
  return this->output.get_dtype().get_elements_per_access();
}

void ParameterSmemImpl::set_instruction_parallel_factor(int _ilp_factor) {
  this->ilp_factor = _ilp_factor;

  for (auto& [tag, auxiliary_impl] : this->auxiliary_impls) {
    auxiliary_impl->set_instruction_parallel_factor(_ilp_factor);
  }
}

std::vector<std::shared_ptr<OpImplBase>>
ParameterSmemImpl::get_available_implementations(
    std::shared_ptr<CUDAContext> cuda_context, InputSpec input_spec,
    Tensor output) {
  std::shared_ptr<ParameterSmemImpl> impl =
      std::make_shared<ParameterSmemImpl>(cuda_context, input_spec, output);
  return {std::static_pointer_cast<OpImplBase>(impl)};
}
}  // namespace op_impl
}  // namespace core
}  // namespace mononn_engine