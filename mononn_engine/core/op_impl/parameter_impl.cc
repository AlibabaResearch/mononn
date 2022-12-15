#include "mononn_engine/core/op_impl/parameter_impl.h"
#include "mononn_engine/core/gpu/buffer_manager.h"
#include "mononn_engine/core/gpu/memory.h"
#include "mononn_engine/core/op_annotation/op_attribute.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
    using Tensor = mononn_engine::core::tensor::Tensor;
    using Dtype = mononn_engine::core::tensor::Dtype;
    using CUDAContext = mononn_engine::core::context::CUDAContext;
    using BufferManager = mononn_engine::core::gpu::BufferManager;
    using Memory = mononn_engine::core::gpu::Memory;
    using TensorSpec = mononn_engine::core::tensor::TensorSpec;
    using OpAttribute = mononn_engine::core::op_annotation::OpAttribute;

    std::string ParameterImpl::generate_with_index_impl() const {
        std::string op_name = this->output.get_name();

        Dtype type = this->output.get_dtype();

        std::string buffer_name = BufferManager::get_buffer_name(op_name) + "_input";

        if ((!this->is_instruction_parallelized() && this->concrete_index_list.size() > 1) ||
            (this->is_instruction_parallelized() && this->ilp_concrete_index_list[0].size() > 1)) { // traced multiple index
            std::stringstream ss;
            if (this->is_instruction_parallelized()) { // ilp
                for (auto const &traced_index : this->ilp_concrete_index_list[0]) {
                    auto const &index = traced_index.index_after_trace;
                    // std::string reuse_op_name = op_name + "_reuse_" + this->get_upstream_ilp_index_trace_node(index, 0);
                    std::string reuse_op_name = op_name + "_reuse_" + traced_index.traced_by;
                    for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor(); ++ilp_id) {
                        ss << type.to_string() << " " << mononn_engine::helpers::get_node_ilp_name(reuse_op_name, ilp_id) << ";\n";
                    }
                }

                for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor(); ++ilp_id) {
                    for (auto const &traced_index : this->ilp_concrete_index_list[ilp_id]) {
                        auto const &index = traced_index.index_after_trace;
                        auto const &pred = traced_index.pred_after_trace;
                        // std::string reuse_op_name = op_name + "_reuse_" + this->get_upstream_ilp_index_trace_node(index, ilp_id);
                        std::string reuse_op_name = op_name + "_reuse_" + traced_index.traced_by;
                        reuse_op_name = mononn_engine::helpers::get_node_ilp_name(reuse_op_name, ilp_id);
                        ss << this->memory_read_wrapper(
                                type,
                                reuse_op_name,
                                buffer_name,
                                index,
                                false,
                                pred);
                    }
                }
            } else { // no ilp
                for (auto const &traced_index : this->concrete_index_list) {
                    auto const &index = traced_index.index_after_trace;
                    auto const &pred = traced_index.pred_after_trace;
                    // std::string reuse_op_name = op_name + "_reuse_" + this->get_upstream_index_trace_node(index);
                    std::string reuse_op_name = op_name + "_reuse_" + traced_index.traced_by;
                    ss << this->memory_read_wrapper(
                            type,
                            reuse_op_name,
                            buffer_name,
                            index,
                            true,
                            pred);
                }
            }

            return ss.str();

        } else { // traced single index
            if (this->is_instruction_parallelized()) { // ilp
                std::stringstream ss;

                for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor(); ++ilp_id) {
                    std::string index = this->ilp_concrete_index_list[ilp_id][0].index_after_trace;
                    std::string pred = this->ilp_concrete_index_list[ilp_id][0].pred_after_trace;
                    std::string ilp_op_name = mononn_engine::helpers::get_node_ilp_name(op_name, ilp_id);

                    ss << this->memory_read_wrapper(
                        type,
                        ilp_op_name,
                        buffer_name,
                        index,
                        true,
                        pred);
                }

                return ss.str();
            } else { // no ilp
                std::string index = this->concrete_index_list[0].index_after_trace;
                std::string pred = this->concrete_index_list[0].pred_after_trace;

                return this->memory_read_wrapper(
                    type,
                    op_name,
                    buffer_name,
                    index,
                    true,
                    pred);
            }
        }
    }

    std::vector<Tensor> ParameterImpl::get_input_tensor() const {
        return {};
    }

    std::vector<Tensor> ParameterImpl::get_output_tensor() const {
        return {this->output};
    }

    int ParameterImpl::get_elements_per_access() const {
        return this->output.get_dtype().get_elements_per_access();
    }

    void ParameterImpl::set_instruction_parallel_factor(int _ilp_factor) {
        this->ilp_factor = _ilp_factor;

        for (auto &[tag, auxiliary_impl] : this->auxiliary_impls) {
            auxiliary_impl->set_instruction_parallel_factor(_ilp_factor);
        }
    }

    std::string ParameterImpl::memory_read_wrapper(Dtype access_type, std::string var_name, std::string src_ptr, std::string offset, bool define_variable, std::string pred) const {
        if (this->has_attribute(OpAttribute::is_parameter_temporal_access)) {
            return Memory::read(
                Memory::AccessFlavor::EXPLICIT_PTX_EVICT_LAST,
                access_type,
                var_name,
                src_ptr,
                offset,
                define_variable,
                pred);
        } else {
            return Memory::read(
                Memory::AccessFlavor::REGULAR,
                access_type,
                var_name,
                src_ptr,
                offset,
                define_variable,
                pred);
        }
    }

    std::vector<std::shared_ptr<OpImplBase>> ParameterImpl::get_available_implementations(
            std::shared_ptr<CUDAContext> cuda_context,
            Tensor output
    ) {
        std::shared_ptr<ParameterImpl> impl = std::make_shared<ParameterImpl>(cuda_context, output);

        return { std::static_pointer_cast<OpImplBase>(impl)};
    }
}
}
}