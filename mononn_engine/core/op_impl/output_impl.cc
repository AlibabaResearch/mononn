#include "mononn_engine/core/op_impl/output_impl.h"
#include "mononn_engine/core/gpu/buffer_manager.h"
#include "mononn_engine/core/gpu/memory.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
    using Tensor = mononn_engine::core::tensor::Tensor;
    using BufferManager = mononn_engine::core::gpu::BufferManager;
    using Memory = mononn_engine::core::gpu::Memory;
    using TensorSpec = mononn_engine::core::tensor::TensorSpec;

    std::string OutputImpl::generate_impl() const {
        std::string node_name = this->input_spec.operand.get_name();
        std::string node_buffer_name = BufferManager::get_buffer_name(node_name) + "_output";
        auto type = this->input_spec.operand.get_dtype();

        if (this->is_instruction_parallelized()) {
            std::stringstream ss;

            for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor(); ++ilp_id) {
                EXPECT_TRUE(this->ilp_concrete_index_list[ilp_id].size() == 1,
                            "Output node should have exactly index, where as " + this->input_spec.operand.get_name() + " have " + std::to_string(this->ilp_concrete_index_list[ilp_id].size()));
                std::string index = this->ilp_concrete_index_list[ilp_id][0].index_before_trace;
                std::string ilp_node_name = mononn_engine::helpers::get_node_ilp_name(node_name, ilp_id);
                ss << Memory::write(Memory::AccessFlavor::REGULAR, type, ilp_node_name, node_buffer_name, index);
            }

            return ss.str();
        } else {
            EXPECT_TRUE(this->concrete_index_list.size() == 1,
                        "Output node should have exactly one index, where as " + this->input_spec.operand.get_name() + " have " + std::to_string(this->get_concrete_index_count()));
            std::string index = this->get_concrete_index(0).index_before_trace;
            return Memory::write(Memory::AccessFlavor::REGULAR, type, node_name, node_buffer_name, index);
        }
    }

    std::vector<Tensor> OutputImpl::get_input_tensor() const {
        return { this->input_spec.operand };
    }

    std::vector<Tensor> OutputImpl::get_output_tensor() const {
        return {};
    }

    int OutputImpl::get_elements_per_access() const {
        return this->input_spec.operand.get_dtype().get_elements_per_access();
    }

    void OutputImpl::set_instruction_parallel_factor(int _ilp_factor) {
        this->ilp_factor = _ilp_factor;

        for (auto &[tag, auxiliary_impl] : this->auxiliary_impls) {
            auxiliary_impl->set_instruction_parallel_factor(_ilp_factor);
        }
    }

    std::vector<std::shared_ptr<OpImplBase>>
    OutputImpl::get_available_implementations(std::shared_ptr<CUDAContext> cuda_context, InputSpec input_spec) {
        std::shared_ptr<OutputImpl> impl = std::make_shared<OutputImpl>(cuda_context, input_spec);

        return { std::static_pointer_cast<OpImplBase>(impl) };
    }
}
}
}