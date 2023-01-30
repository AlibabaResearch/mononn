#include "mononn_engine/core/op_impl/dynamic_slice_impl.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
using Tensor = mononn_engine::core::tensor::Tensor;
using Dtype = mononn_engine::core::tensor::Dtype;
using TensorSpec = mononn_engine::core::tensor::TensorSpec;

std::string DynamicSliceImpl::generate_impl() const {
  std::string operand_name = this->input_spec.operands[0].get_name();
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

int DynamicSliceImpl::get_elements_per_access() const {
  return this->output.get_dtype().get_elements_per_access();
}

std::vector<Tensor> DynamicSliceImpl::get_input_tensor() const {
  return this->input_spec.operands;
}

std::vector<Tensor> DynamicSliceImpl::get_output_tensor() const {
  return {this->output};
}

void DynamicSliceImpl::set_instruction_parallel_factor(int _ilp_factor) {
  this->ilp_factor = _ilp_factor;
  for (auto& [tag, auxiliary_impl] : this->auxiliary_impls) {
    auxiliary_impl->set_instruction_parallel_factor(_ilp_factor);
  }
}

std::vector<std::shared_ptr<OpImplBase>>
DynamicSliceImpl::get_available_implementations(
    std::shared_ptr<CUDAContext> cuda_context,
    DynamicSliceImpl::InputSpec input_spec, Tensor output) {
  std::shared_ptr<DynamicSliceImpl> slice_impl =
      std::make_shared<DynamicSliceImpl>(cuda_context, input_spec, output);

  return {std::static_pointer_cast<OpImplBase>(slice_impl)};
}
}  // namespace op_impl
}  // namespace core
}  // namespace mononn_engine