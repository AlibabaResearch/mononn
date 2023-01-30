#include "mononn_engine/core/op_impl/reshape_impl.h"

#include "mononn_engine/core/tensor/dtype.h"
#include "mononn_engine/core/tensor/tensor.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
using Tensor = mononn_engine::core::tensor::Tensor;
using Dtype = mononn_engine::core::tensor::Dtype;
using TensorSpec = mononn_engine::core::tensor::TensorSpec;

std::string ReshapeImpl::generate_impl() const {
  std::string operand_name = this->input_spec.operand.get_name();
  std::string node_name = this->output.get_name();
  Dtype type = this->output.get_dtype();

  std::stringstream ss;

  if (this->is_instruction_parallelized()) {
    for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor();
         ++ilp_id) {
      ss << type.to_string() << " "
         << mononn_engine::helpers::get_node_ilp_name(node_name, ilp_id)
         << " = "
         << mononn_engine::helpers::get_node_ilp_name(operand_name, ilp_id)
         << ";\n";
    }
  } else {
    ss << type.to_string() << " " << node_name << " = " << operand_name
       << ";\n";
  }

  return ss.str();
}

std::vector<Tensor> ReshapeImpl::get_input_tensor() const {
  return {this->input_spec.operand};
}

std::vector<Tensor> ReshapeImpl::get_output_tensor() const {
  return {this->output};
}

int ReshapeImpl::get_elements_per_access() const {
  return this->output.get_dtype().get_elements_per_access();
}

void ReshapeImpl::set_instruction_parallel_factor(int _ilp_factor) {
  this->ilp_factor = _ilp_factor;

  for (auto& [tag, auxiliary_impl] : this->auxiliary_impls) {
    auxiliary_impl->set_instruction_parallel_factor(_ilp_factor);
  }
}

std::vector<std::shared_ptr<OpImplBase>>
ReshapeImpl::get_available_implementations(
    std::shared_ptr<CUDAContext> cuda_context,
    ReshapeImpl::InputSpec input_spec, ReshapeImpl::Tensor output) {
  std::shared_ptr<ReshapeImpl> reshape_impl =
      std::make_shared<ReshapeImpl>(cuda_context, input_spec, output);

  return {std::static_pointer_cast<OpImplBase>(reshape_impl)};
}
}  // namespace op_impl
}  // namespace core
}  // namespace mononn_engine