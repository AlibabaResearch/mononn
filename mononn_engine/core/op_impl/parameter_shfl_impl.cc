#include <sstream>
#include "mononn_engine/core/op_impl/parameter_shfl_impl.h"
#include "mononn_engine/helpers/string_helpers.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
    using Tensor = mononn_engine::core::tensor::Tensor;
    using TensorSpec = mononn_engine::core::tensor::TensorSpec;

    std::string ParameterShflImpl::generate_impl() const {
        std::stringstream ss;
        auto type = this->output.get_dtype();

        if (this->is_instruction_parallelized()) {
            for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor(); ++ilp_id) {
                ss << type.to_string() << " " << mononn_engine::helpers::get_node_ilp_name(this->output.get_name(), ilp_id) << " = ";
                ss << this->input_spec.operand.get_name() << ";";
                ss << "\n";
            }
        } else {
            ss << type.to_string() << " " << this->output.get_name() << " = ";
            ss << this->input_spec.operand.get_name() << ";";
            ss << "\n";
        }

        return ss.str();
    }

    std::vector<Tensor> ParameterShflImpl::get_input_tensor() const {
        return { this->input_spec.operand };
    }

    std::vector<Tensor> ParameterShflImpl::get_output_tensor() const {
        return { this->output };
    }

    int ParameterShflImpl::get_elements_per_access() const {
        return this->output.get_dtype().get_elements_per_access();
    }

    void ParameterShflImpl::set_instruction_parallel_factor(int _ilp_factor) {
        this->ilp_factor = _ilp_factor;

        for (auto &[tag, auxiliary_impl] : this->auxiliary_impls) {
            auxiliary_impl->set_instruction_parallel_factor(_ilp_factor);
        }
    }

    std::vector<std::shared_ptr<OpImplBase>>
    ParameterShflImpl::get_available_implementations(
        std::shared_ptr<CUDAContext> cuda_context,
        InputSpec input_spec,
        Tensor output) {
        std::shared_ptr<ParameterShflImpl> impl = std::make_shared<ParameterShflImpl>(cuda_context, input_spec, output);
        return { std::static_pointer_cast<OpImplBase>(impl) };
    }
}
}
}