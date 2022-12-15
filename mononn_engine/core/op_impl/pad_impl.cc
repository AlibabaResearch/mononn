#include <sstream>
#include "mononn_engine/core/op_impl/pad_impl.h"
#include "mononn_engine/core/tensor/dtype.h"
#include "mononn_engine/core/gpu/buffer_manager.h"
#include "mononn_engine/core/gpu/memory.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
    using Dtype = mononn_engine::core::tensor::Dtype;
    using Tensor = mononn_engine::core::tensor::Tensor;
    using OpImplBase = mononn_engine::core::op_impl::OpImplBase;
    using BufferManager = mononn_engine::core::gpu::BufferManager;
    using Memory = mononn_engine::core::gpu::Memory;
    using TensorSpec = mononn_engine::core::tensor::TensorSpec;

    std::string PadImpl::generate_impl() const {
        std::string operand_name = this->input_spec.operand.get_name();
        std::string node_name = this->output.get_name();
        Dtype type = this->output.get_dtype();

        std::stringstream ss;

        if (this->is_instruction_parallelized()) {
             for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor(); ++ilp_id) {
                 ss << type.to_string() << " " << mononn_engine::helpers::get_node_ilp_name(node_name, ilp_id) << " = " <<
                    mononn_engine::helpers::get_node_ilp_name(operand_name, ilp_id) << ";\n";
             }
        } else {
            ss << type.to_string() << " " << node_name << " = " << operand_name << ";\n";
        }

        return ss.str();
    }

    std::string PadImpl::generate_with_index_impl() const {
        std::string operand_name = this->input_spec.operand.get_name();
        std::string buffer_name = BufferManager::get_buffer_name(operand_name);
        std::string node_name = this->output.get_name();
        Dtype type = this->output.get_dtype();
        std::string index = this->concrete_index_list[0].index_after_trace;
        std::string pred = this->concrete_index_list[0].pred_after_trace;

        std::stringstream ss;
        ss << Memory::read(Memory::AccessFlavor::REGULAR, type, node_name, buffer_name, index, false, pred);

        return ss.str();
    }

    int PadImpl::get_elements_per_access() const {
        return this->output.get_dtype().get_elements_per_access();
    }

    std::vector<Tensor> PadImpl::get_input_tensor() const {
        return {
            this->input_spec.operand
        };
    }

    std::vector<Tensor> PadImpl::get_output_tensor() const {
        return {
            this->output
        };
    }

//    std::string PadImpl::generate_if_statement_cond(std::vector<std::string> multi_index) const {
//        EXPECT_TRUE(multi_index.size() == this->input_spec.padding_low.size(), "Rank mismatch");
//        EXPECT_TRUE(multi_index.size() == this->input_spec.padding_high.size(), "Rank mismatch");
//
//        int rank = (int)multi_index.size();
//
//        std::string result;
//
//        for (int idx = 0; idx < rank; ++idx) {
//            std::string cond;
//
//            if (this->input_spec.padding_low[idx] != 0) {
//                cond += mononn_engine::helpers::string_format("(%s >= %s)", multi_index[idx].c_str(), std::to_string(this->input_spec.padding_low[idx]).c_str());
//            }
//
//            if (this->input_spec.padding_high[idx] != 0) {
//                if (cond.length() != 0) cond += " && ";
//                cond += mononn_engine::helpers::string_format("(%s < %s)", multi_index[idx].c_str(),
//                                                         std::to_string(this->output.get_shape(idx) - this->input_spec.padding_high[idx]).c_str());
//            }
//
//            if (cond.length() != 0) {
//                if (result.length() != 0) result += " && ";
//
//                result += "(" + cond + ")";
//            }
//        }
//
//        return result;
//    }
//
//    std::string PadImpl::generate_if_statement_begin(std::vector<std::string> multi_index) const {
//        std::stringstream ss;
//        ss << "if (";
//        ss << this->generate_if_statement_cond(multi_index) << ") {\n";
//
//        return ss.str();
//    }
//
//    std::string PadImpl::generate_if_statement_end() const {
//        return "}\n";
//    }
//
//    std::string PadImpl::generate_else() const {
//        std::stringstream ss;
//        ss << "else {\n";
//        ss << this->output.get_name() << " = ";
//
//        if (this->output.get_dtype() == Dtype::from_string("float16")) {
//            ss << "half(" << this->input_spec.padding_value << ")" << ";\n";
//        } else {
//            ss << this->input_spec.padding_value<< ";\n";
//        }
//
//        ss << "}\n";
//
//        return ss.str();
//    }

    void PadImpl::set_instruction_parallel_factor(int _ilp_factor) {
        this->ilp_factor = _ilp_factor;

        for (auto &[tag, auxiliary_impl] : this->auxiliary_impls) {
            auxiliary_impl->set_instruction_parallel_factor(_ilp_factor);
        }
    }

    std::vector<std::shared_ptr<OpImplBase>>
    PadImpl::get_available_implementations(std::shared_ptr<CUDAContext> cuda_context, PadImpl::InputSpec input_spec,
                                           Tensor output) {
        std::shared_ptr<PadImpl> pad_impl = std::make_shared<PadImpl>(cuda_context, input_spec, output);
        return {
            std::static_pointer_cast<OpImplBase>(pad_impl)
        };
    }
}
}
}