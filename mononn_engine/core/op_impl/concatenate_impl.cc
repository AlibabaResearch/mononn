#include "mononn_engine/core/op_impl/concatenate_impl.h"

#include "mononn_engine/core/context/index_transform.h"
#include "mononn_engine/core/gpu/functor.h"
#include "mononn_engine/core/tensor/dtype.h"
#include "mononn_engine/core/tensor/tensor.h"
#include "mononn_engine/core/tensor/tensor_spec.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
using Dtype = mononn_engine::core::tensor::Dtype;
using Tensor = mononn_engine::core::tensor::Tensor;
using TensorSpec = mononn_engine::core::tensor::TensorSpec;
using OpType = mononn_engine::core::op::OpType;
using Functor = mononn_engine::core::gpu::Functor;
using IndexTransform = mononn_engine::core::context::IndexTransform;
using TensorShape = mononn_engine::core::tensor::TensorShape;

std::string ConcatenateImpl::generate_with_index_impl() const {
  auto type = this->output.get_dtype();
  auto primitive_type = type.get_primitive_type();
  std::string node_name = this->output.get_name();
  auto shape = this->output.get_shape();
  std::vector<TensorShape> operand_shape;

  for (auto const& operand : this->input_spec.operands) {
    operand_shape.push_back(operand.get_shape());
  }

  std::stringstream ss;

  if (this->is_instruction_parallelized()) {
    for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor();
         ++ilp_id) {
      EXPECT_TRUE(this->ilp_concrete_index_list[ilp_id].size() == 1,
                  "Multiple traced index");
      std::string offset_index =
          this->ilp_concrete_index_list[ilp_id][0].index_before_trace;
      std::vector<std::string> per_dim_index =
          IndexTransform::offset_to_multi_index(shape, offset_index);
      int concat_dimension = this->input_spec.dimension;
      std::string concat_dimension_index = per_dim_index[concat_dimension];
      std::vector<std::string> predictions;

      for (int operand_id = 0;
           operand_id < (int)this->input_spec.operands.size() - 1;
           ++operand_id) {
        std::string concat_dimension_range;

        for (int idx = 0; idx <= operand_id; ++idx) {
          if (idx == 0)
            concat_dimension_range = std::to_string(
                this->input_spec.operands[idx].get_shape(concat_dimension));
          else
            concat_dimension_range =
                concat_dimension_range + " + " +
                std::to_string(
                    this->input_spec.operands[idx].get_shape(concat_dimension));
        }

        predictions.push_back(mononn_engine::helpers::string_format(
            "(%s < %s)", concat_dimension_index.c_str(),
            concat_dimension_range.c_str()));
      }

      FunctionInvocation concat(mononn_engine::helpers::string_format(
          "cutlass::Concat<%s, %d>::do_concat",
          primitive_type.to_string().c_str(), type.get_elements_per_access()));

      for (int operand_id = 0;
           operand_id < (int)this->input_spec.operands.size() - 1;
           ++operand_id) {
        concat.add_arg(predictions[operand_id]);
        concat.add_arg(mononn_engine::helpers::get_node_ilp_name(
            this->input_spec.operands[operand_id].get_name(), ilp_id));
      }

      concat.add_arg(mononn_engine::helpers::get_node_ilp_name(
          this->input_spec.operands.back().get_name(), ilp_id));

      ss << type.to_string() << " "
         << mononn_engine::helpers::get_node_ilp_name(node_name, ilp_id)
         << " = " << concat.to_string() << ";\n";
    }
  } else {
    EXPECT_TRUE(this->concrete_index_list.size() == 1, "Multiple traced index");

    std::string offset_index = this->concrete_index_list[0].index_before_trace;
    std::vector<std::string> per_dim_index =
        IndexTransform::offset_to_multi_index(shape, offset_index);
    int concat_dimension = this->input_spec.dimension;
    std::string concat_dimension_index = per_dim_index[concat_dimension];
    std::vector<std::string> predictions;

    for (int operand_id = 0;
         operand_id < (int)this->input_spec.operands.size() - 1; ++operand_id) {
      std::string concat_dimension_range;

      for (int idx = 0; idx <= operand_id; ++idx) {
        if (idx == 0)
          concat_dimension_range = std::to_string(
              this->input_spec.operands[idx].get_shape(concat_dimension));
        else
          concat_dimension_range =
              concat_dimension_range + " + " +
              std::to_string(
                  this->input_spec.operands[idx].get_shape(concat_dimension));
      }

      predictions.push_back(mononn_engine::helpers::string_format(
          "(%s < %s)", concat_dimension_index.c_str(),
          concat_dimension_range.c_str()));
    }

    FunctionInvocation concat(mononn_engine::helpers::string_format(
        "cutlass::Concat<%s, %d>::do_concat",
        primitive_type.to_string().c_str(), type.get_elements_per_access()));

    for (int operand_id = 0;
         operand_id < (int)this->input_spec.operands.size() - 1; ++operand_id) {
      concat.add_arg(predictions[operand_id]);
      concat.add_arg(this->input_spec.operands[operand_id].get_name());
    }

    concat.add_arg(this->input_spec.operands.back().get_name());

    ss << type.to_string() << " " << node_name << " = " << concat.to_string()
       << ";\n";
  }

  return ss.str();
}

int ConcatenateImpl::get_elements_per_access() const {
  return this->output.get_dtype().get_elements_per_access();
}

std::vector<Tensor> ConcatenateImpl::get_input_tensor() const {
  return this->input_spec.operands;
}

std::vector<Tensor> ConcatenateImpl::get_output_tensor() const {
  return {this->output};
}

void ConcatenateImpl::set_instruction_parallel_factor(int _ilp_factor) {
  this->ilp_factor = _ilp_factor;

  for (auto& [tag, auxiliary_impl] : this->auxiliary_impls) {
    auxiliary_impl->set_instruction_parallel_factor(_ilp_factor);
  }
}

std::vector<std::shared_ptr<OpImplBase>>
ConcatenateImpl::get_available_implementations(
    std::shared_ptr<CUDAContext> cuda_context, InputSpec input_spec,
    Tensor output) {
  std::shared_ptr<ConcatenateImpl> impl =
      std::make_shared<ConcatenateImpl>(cuda_context, input_spec, output);

  return {std::static_pointer_cast<OpImplBase>(impl)};
}
}  // namespace op_impl
}  // namespace core
}  // namespace mononn_engine
