#include "mononn_engine/core/op_impl/output_reg_impl.h"
#include "mononn_engine/core/gpu/memory.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
    using Tensor = mononn_engine::core::tensor::Tensor;
    using TensorSpec = mononn_engine::core::tensor::TensorSpec;
    using Memory = mononn_engine::core::gpu::Memory;
    
    std::string OutputRegImpl::generate_impl() const {
        auto type = this->input_spec.operand.get_dtype();

        if (this->is_instruction_parallelized()) {
            std::stringstream ss;

            for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor(); ++ilp_id) {
                std::string node_name = this->input_spec.operand.get_name();
                // std::string store_reg_index = mononn_engine::helpers::string_format("(%s) + (%s)",
                //         this->input_spec.step_id.c_str(),
                //         std::to_string(ilp_id).c_str());

                // ss << mononn_engine::helpers::string_format("%s[%s] = %s;\n",
                //                                         this->input_spec.reg_buffer_name.c_str(),
                //                                         store_reg_index.c_str(),
                //                                         mononn_engine::helpers::get_node_ilp_name(node_name, ilp_id).c_str());

                std::string index = mononn_engine::helpers::string_format("%s + %d", this->input_spec.step_id.c_str(), ilp_id);
                ss << Memory::write(
                    Memory::AccessFlavor::STRONG_TYPED, 
                    type, 
                    mononn_engine::helpers::get_node_ilp_name(node_name, ilp_id),
                    this->input_spec.reg_buffer_name,
                    index);
            }

            return ss.str();
        } else {
            std::string node_name = this->input_spec.operand.get_name();

            return Memory::write(Memory::AccessFlavor::STRONG_TYPED, type, node_name, this->input_spec.reg_buffer_name, this->input_spec.step_id);
            // return mononn_engine::helpers::string_format("%s[%s] = %s;\n",
            //         this->input_spec.reg_buffer_name.c_str(),
            //         this->input_spec.step_id.c_str(),
            //         node_name.c_str());
        }
    }

    std::vector<Tensor> OutputRegImpl::get_input_tensor() const {
        return { this->input_spec.operand };
    }

    std::vector<Tensor> OutputRegImpl::get_output_tensor() const {
        return { };
    }

    int OutputRegImpl::get_elements_per_access() const {
        return this->input_spec.operand.get_dtype().get_elements_per_access();
    }

    bool OutputRegImpl::need_pre_inner_loop_generation() const {
        return true;
    }

    void OutputRegImpl::set_instruction_parallel_factor(int _ilp_factor) {
        this->ilp_factor = _ilp_factor;

        for (auto &[tag, auxiliary_impl] : this->auxiliary_impls) {
            auxiliary_impl->set_instruction_parallel_factor(_ilp_factor);
        }
    }

    std::string OutputRegImpl::generate_pre_inner_loop() const {
        auto type = this->input_spec.operand.get_dtype();

        return mononn_engine::helpers::string_format("%s %s[%s];\n",
            type.to_string().c_str(),
            this->input_spec.reg_buffer_name.c_str(),
            this->input_spec.step_cnt.c_str());

        // if (this->is_instruction_parallelized()) {
        //     std::stringstream ss;

        //     for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor(); ++ilp_id) {
        //         ss << mononn_engine::helpers::string_format("%s %s[%s];\n",
        //                 type.to_string().c_str(),
        //                 mononn_engine::helpers::get_node_ilp_name(this->input_spec.reg_buffer_name, ilp_id).c_str(),
        //                 this->input_spec.step_cnt.c_str());
        //     }

        //     return ss.str();
        // } else {
        //     return mononn_engine::helpers::string_format("%s %s[%s];\n",
        //     type.to_string().c_str(),
        //     this->input_spec.reg_buffer_name.c_str(),
        //     this->input_spec.step_cnt.c_str());
        // }
    }
}
}
}