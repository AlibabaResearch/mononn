#include "mononn_engine/core/op_impl/dynamic_update_slice_impl.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
    using Tensor = mononn_engine::core::tensor::Tensor;
    using Dtype = mononn_engine::core::tensor::Dtype;
    using TensorSpec = mononn_engine::core::tensor::TensorSpec;

    std::string DynamicUpdateSliceImpl::generate_with_index_impl() const {
        std::string operand_name = this->input_spec.operands[0].get_name();
        std::string update_name = this->input_spec.operands[1].get_name();
        std::string node_name = this->output.get_name();

        Dtype type = this->output.get_dtype();

        std::stringstream ss;

        if (this->is_instruction_parallelized()) {
            for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor(); ++ilp_id) {
                ss << type.to_string() << " " << mononn_engine::helpers::get_node_ilp_name(node_name, ilp_id)
                   << " = " << mononn_engine::helpers::get_node_ilp_name(operand_name, ilp_id) << ";\n";

                ss << mononn_engine::helpers::string_format("%s %s = %s ? %s : %s;\n", 
                    type.to_string().c_str(), 
                    mononn_engine::helpers::get_node_ilp_name(node_name, ilp_id).c_str(), 
                    this->ilp_concrete_index_list[ilp_id][0].pred_after_trace.c_str(),
                    mononn_engine::helpers::get_node_ilp_name(operand_name, ilp_id), 
                    mononn_engine::helpers::get_node_ilp_name(update_name, ilp_id).c_str());
            }
        } else {
            ss << mononn_engine::helpers::string_format("%s %s = %s ? %s : %s;\n", 
            type.to_string().c_str(), node_name.c_str(), this->concrete_index_list[0].pred_after_trace.c_str(), operand_name.c_str(), update_name.c_str());
        }

        return ss.str();
    }

    int DynamicUpdateSliceImpl::get_elements_per_access() const {
        return this->output.get_dtype().get_elements_per_access();
    }

    std::vector<Tensor> DynamicUpdateSliceImpl::get_input_tensor() const {
        return this->input_spec.operands;
    }

    std::vector<Tensor> DynamicUpdateSliceImpl::get_output_tensor() const {
        return { this->output };
    }

    void DynamicUpdateSliceImpl::set_instruction_parallel_factor(int _ilp_factor) {
        this->ilp_factor = _ilp_factor;
        for (auto &[tag, auxiliary_impl] : this->auxiliary_impls) {
            auxiliary_impl->set_instruction_parallel_factor(_ilp_factor);
        }
    }

    std::vector<std::shared_ptr<OpImplBase>>
    DynamicUpdateSliceImpl::get_available_implementations(std::shared_ptr<CUDAContext> cuda_context, DynamicUpdateSliceImpl::InputSpec input_spec,
                                             Tensor output) {
        std::shared_ptr<DynamicUpdateSliceImpl> slice_impl =
                std::make_shared<DynamicUpdateSliceImpl>(cuda_context, input_spec, output);

        return {
            std::static_pointer_cast<OpImplBase>(slice_impl)
        };
    }
}
}
}