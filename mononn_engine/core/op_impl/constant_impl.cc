#include "mononn_engine/core/op_impl/constant_impl.h"

#include "mononn_engine/core/gpu/buffer_manager.h"
#include "mononn_engine/core/gpu/limits.h"
#include "mononn_engine/core/gpu/memory.h"
#include "mononn_engine/core/tensor/dtype.h"
#include "mononn_engine/helpers/string_helpers.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
using Tensor = mononn_engine::core::tensor::Tensor;
using Memory = mononn_engine::core::gpu::Memory;
using BufferManager = mononn_engine::core::gpu::BufferManager;
using Dtype = mononn_engine::core::tensor::Dtype;
using CUDAContext = mononn_engine::core::context::CUDAContext;
using Limits = mononn_engine::core::gpu::Limits;
using TensorSpec = mononn_engine::core::tensor::TensorSpec;

std::string ConstantImpl::generate_impl() const {
  std::string op_name = this->output.get_name();
  Dtype type = this->output.get_dtype();

  if (this->output.element_count() != 1) {
    LOG(FATAL) << this->output.get_name()
               << " is not scalar, need index to fetch value";
  }

  std::stringstream ss;
  ss << "const " << type.to_string() << " " << op_name << " = ";

  if (this->input_spec.value == "inf") {
    ss << Limits::get_max_positive(type.get_primitive_type()) << ";";
  } else if (this->input_spec.value == "-inf") {
    ss << Limits::get_min_negative(type.get_primitive_type()) << ";";
  } else if (type.get_primitive_type() == Dtype::FLOAT16) {
    ss << mononn_engine::helpers::string_format("half(%s);",
                                                this->input_spec.value.c_str());
  } else {
    ss << mononn_engine::helpers::string_format("%s;",
                                                this->input_spec.value.c_str());
  }

  ss << "\n";

  if (this->is_instruction_parallelized()) {
    for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor();
         ++ilp_id) {
      ss << "const " << type.to_string() << " &"
         << mononn_engine::helpers::get_node_ilp_name(op_name, ilp_id) << " = "
         << op_name << ";\n";
    }
  }

  return ss.str();
}

std::string ConstantImpl::generate_with_index_impl() const {
  LOG(FATAL)
      << "Deprecated, all non-scalar constant should be read as parameter.";

  std::string op_name = this->output.get_name();

  Dtype type = this->output.get_dtype();
  std::string buffer_name = BufferManager::get_buffer_name(op_name);

  if ((!this->is_instruction_parallelized() &&
       this->concrete_index_list.size() > 1) ||
      (this->is_instruction_parallelized() &&
       this->ilp_concrete_index_list[0].size() >
           1)) {  // need operand reuse mask
    std::stringstream ss;
    if (this->is_instruction_parallelized()) {
      for (auto const& traced_index : this->ilp_concrete_index_list[0]) {
        auto const& index = traced_index.index_after_trace;
        std::string reuse_op_name =
            op_name + "_reuse_" +
            this->get_upstream_ilp_index_trace_node(index, 0);
        ss << type.to_string() << " " << reuse_op_name << ";\n";
      }

      for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor();
           ++ilp_id) {
        for (auto const& traced_index : this->ilp_concrete_index_list[ilp_id]) {
          auto const& index = traced_index.index_after_trace;
          auto const& pred = traced_index.pred_after_trace;
          std::string reuse_op_name =
              op_name + "_reuse_" +
              this->get_upstream_ilp_index_trace_node(index, ilp_id);
          reuse_op_name = reuse_op_name + "[" + std::to_string(ilp_id) + "]";
          ss << Memory::read(Memory::AccessFlavor::REGULAR,
                             type.get_primitive_type(), reuse_op_name,
                             buffer_name, index, false, pred);
        }
      }
    } else {
      for (auto const& traced_index : this->concrete_index_list) {
        auto const& index = traced_index.index_after_trace;
        auto const& pred = traced_index.pred_after_trace;
        std::string reuse_op_name =
            op_name + "_reuse_" + this->get_upstream_index_trace_node(index);
        ss << Memory::read(Memory::AccessFlavor::REGULAR, type, reuse_op_name,
                           buffer_name, index, true, pred);
      }
    }

    return ss.str();

  } else {
    if (this->is_instruction_parallelized()) {
      std::stringstream ss;
      ss << type.to_string() << " " << op_name << ";\n";

      for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor();
           ++ilp_id) {
        std::string index =
            this->ilp_concrete_index_list[ilp_id][0].index_after_trace;
        std::string pred =
            this->ilp_concrete_index_list[ilp_id][0].pred_after_trace;
        std::string ilp_op_name = op_name + "[" + std::to_string(ilp_id) + "]";
        ss << Memory::read(Memory::AccessFlavor::REGULAR,
                           type.get_primitive_type(), ilp_op_name, buffer_name,
                           index, false, pred);
      }

      return ss.str();
    } else {
      std::string index = this->concrete_index_list[0].index_after_trace;
      std::string pred = this->concrete_index_list[0].pred_after_trace;
      return Memory::read(Memory::AccessFlavor::REGULAR, type, op_name,
                          buffer_name, index, true, pred);
    }
  }
}

std::vector<Tensor> ConstantImpl::get_input_tensor() const { return {}; }

std::vector<Tensor> ConstantImpl::get_output_tensor() const {
  return {this->output};
}

int ConstantImpl::get_elements_per_access() const {
  return this->output.get_dtype().get_elements_per_access();
}

std::vector<std::shared_ptr<OpImplBase>>
ConstantImpl::get_available_implementations(
    std::shared_ptr<CUDAContext> cuda_context, InputSpec input_spec,
    Tensor output) {
  std::shared_ptr<ConstantImpl> constant_impl =
      std::make_shared<ConstantImpl>(cuda_context, input_spec, output);

  return {std::static_pointer_cast<OpImplBase>(constant_impl)};
}

void ConstantImpl::set_instruction_parallel_factor(int _ilp_factor) {
  this->ilp_factor = _ilp_factor;

  for (auto& [tag, auxiliary_impl] : this->auxiliary_impls) {
    auxiliary_impl->set_instruction_parallel_factor(_ilp_factor);
  }
}
}  // namespace op_impl
}  // namespace core
}  // namespace mononn_engine