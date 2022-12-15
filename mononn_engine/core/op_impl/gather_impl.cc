#include "mononn_engine/core/op_impl/gather_impl.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
    using Tensor = mononn_engine::core::tensor::Tensor;
    using CUDAContext = mononn_engine::core::context::CUDAContext;
    using InputSpec = GatherImpl::InputSpec;
    using TensorSpec = mononn_engine::core::tensor::TensorSpec;
    
    std::string GatherImpl::generate_with_index_impl() const {
        std::stringstream ss;

        if (this->is_instruction_parallelized()) {
            for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor(); ++ilp_id) {
                ss << this->output.get_dtype().to_string() << " " << mononn_engine::helpers::get_node_ilp_name(this->output.get_name(), ilp_id) <<
                   " = " << mononn_engine::helpers::get_node_ilp_name(this->input_spec.operand.get_name(), ilp_id) << ";\n";
            }
        } else {
            ss << this->output.get_dtype().to_string() << " " << this->output.get_name() <<
               " = " << this->input_spec.operand.get_name() << ";\n";
        }

        return ss.str();
    }

    std::vector<Tensor> GatherImpl::get_input_tensor() const {
        return {this->input_spec.operand, this->input_spec.start_indices};
    }

    std::vector<Tensor> GatherImpl::get_output_tensor() const {
        return {this->output};
    }

    int GatherImpl::get_elements_per_access() const {
        return this->input_spec.operand.get_dtype().get_elements_per_access();
    }

    void GatherImpl::set_instruction_parallel_factor(int _ilp_factor) {
        this->ilp_factor = _ilp_factor;

        for (auto &[tag, auxiliary_impl] : this->auxiliary_impls) {
            auxiliary_impl->set_instruction_parallel_factor(_ilp_factor);
        }
    }

    std::vector<std::shared_ptr<OpImplBase>> GatherImpl::get_available_implementations(
            std::shared_ptr<CUDAContext> cuda_context, 
            InputSpec input_spec, 
            Tensor output
    ) {
        std::shared_ptr<GatherImpl> gather_impl = std::make_shared<GatherImpl>(cuda_context, input_spec, output);

        return { std::static_pointer_cast<OpImplBase>(gather_impl) };
    }
}
}
}