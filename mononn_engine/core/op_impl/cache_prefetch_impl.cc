#include "mononn_engine/core/tensor/dtype.h"
#include "mononn_engine/core/gpu/buffer_manager.h"
#include "mononn_engine/core/gpu/memory.h"
#include "mononn_engine/core/op_impl/cache_prefetch_impl.h"
#include "mononn_engine/core/op_annotation/op_attribute.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
    using Dtype = mononn_engine::core::tensor::Dtype;
    using Memory = mononn_engine::core::gpu::Memory;
    using BufferManager = mononn_engine::core::gpu::BufferManager;
    using OpAttribute = mononn_engine::core::op_annotation::OpAttribute;

    std::string CachePrefetchImpl::generate_with_index_impl() const {
        std::string op_name = this->input_spec.operand.get_name();

        Dtype type = this->input_spec.operand.get_dtype();
        std::string buffer_name = BufferManager::get_buffer_name(op_name);

        if ((!this->is_instruction_parallelized() && this->concrete_index_list.size() > 1) ||
            (this->is_instruction_parallelized() && this->ilp_concrete_index_list[0].size() > 1)) { // traced multiple index
            std::stringstream ss;
            if (this->is_instruction_parallelized()) { // ilp
                for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor(); ++ilp_id) {
                    for (auto const &traced_index : this->ilp_concrete_index_list[ilp_id]) {
                        auto const &index = traced_index.index_after_trace;
                        auto const &pred = traced_index.pred_after_trace;
                        std::string prefetch_predicate = mononn_engine::helpers::string_format("(%s) && (%s)", pred.c_str(), this->get_attribute(OpAttribute::prefetch_predicate).c_str());
                        ss << Memory::prefetch_l1(type, buffer_name, index, prefetch_predicate);
                    }
                }
            } else { // no ilp
                for (auto const &traced_index : this->concrete_index_list) {
                    auto const &index = traced_index.index_after_trace;
                    auto const &pred = traced_index.pred_after_trace;
                    std::string prefetch_predicate = mononn_engine::helpers::string_format("(%s) && (%s)", pred.c_str(), this->get_attribute(OpAttribute::prefetch_predicate).c_str());
                    ss << Memory::prefetch_l1(type, buffer_name, index, prefetch_predicate);
                }
            }

            return ss.str();
        } else { // traced single index
            if (this->is_instruction_parallelized()) { // ilp
                std::stringstream ss;
                for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor(); ++ilp_id) {
                    std::string index = this->ilp_concrete_index_list[ilp_id][0].index_after_trace;
                    std::string pred = this->ilp_concrete_index_list[ilp_id][0].pred_after_trace;
                    std::string prefetch_predicate = mononn_engine::helpers::string_format("(%s) && (%s)", pred.c_str(), this->get_attribute(OpAttribute::prefetch_predicate).c_str());
                    ss << Memory::prefetch_l1(type, buffer_name, index, prefetch_predicate);
                }

                return ss.str();
            } else { // no ilp
                std::string index = this->concrete_index_list[0].index_after_trace;
                std::string pred = this->concrete_index_list[0].pred_after_trace;
                std::string prefetch_predicate = mononn_engine::helpers::string_format("(%s) && (%s)", pred.c_str(), this->get_attribute(OpAttribute::prefetch_predicate).c_str());
                return Memory::prefetch_l1(type, buffer_name, index, prefetch_predicate);
            }
        }
    }

    int CachePrefetchImpl::get_elements_per_access() const {
        this->input_spec.operand.get_dtype().get_elements_per_access();
    }

    std::vector<Tensor> CachePrefetchImpl::get_input_tensor() const {
        return { this->input_spec.operand };
    }

    std::vector<Tensor> CachePrefetchImpl::get_output_tensor() const {
        return {};
    }

    void CachePrefetchImpl::set_instruction_parallel_factor(int _ilp_factor) {
        this->ilp_factor = _ilp_factor;

        for (auto &[tag, auxiliary_impl] : this->auxiliary_impls) {
            auxiliary_impl->set_instruction_parallel_factor(_ilp_factor);
        }
    }

    void CachePrefetchImpl::instantiate_concrete_index_impl(
        const std::vector<SymbolicIndexStamp> &symbolic_index_list,
        const std::map<std::string, std::string> &params,
        const std::string &loop_stride) {

        std::map<std::string, std::string> prefetch_params;

        for (auto const &[key, value] : params) {
            if (key == "linear_index") {
                prefetch_params[key] = mononn_engine::helpers::string_format("(%s) + (%s)", value.c_str(), loop_stride.c_str());
            } else {
                prefetch_params[key] = value;
            }
        }

        prefetch_params["ilp_variable_suffix"] = "";

        this->concrete_index_list.clear();

        for (auto const &symbolic_index : symbolic_index_list) {
            this->concrete_index_list.push_back(symbolic_index.instantiate(prefetch_params));
        }
    }

    void CachePrefetchImpl::instantiate_ilp_concrete_index_impl(
        const std::vector<SymbolicIndexStamp> &symbolic_index_list,
        const std::map<std::string, std::string> &params,
        const std::string &loop_stride,
        const std::string &ilp_stride) {
        
        this->ilp_concrete_index_list.clear();
        this->ilp_concrete_index_list.resize(this->get_instruction_parallel_factor());

        for (int ilp_id = 0; ilp_id < this->get_instruction_parallel_factor(); ++ilp_id) {

            std::map<std::string, std::string> ilp_prefetch_params;
            for (auto const& [key, value] : params) {
                if (key == "linear_index") {
                    std::pair<std::string, std::string> new_param;
                    ilp_prefetch_params[key] = mononn_engine::helpers::string_format("((%s) + (%s) + (%s * %d))", value.c_str(), loop_stride.c_str(), ilp_stride.c_str(), ilp_id);
                } else {
                    ilp_prefetch_params[key] = value;
                }
            }

            ilp_prefetch_params["ilp_variable_suffix"] =  "__i" + std::to_string(ilp_id);

            for (auto const &symbolic_index : symbolic_index_list) {
                this->ilp_concrete_index_list[ilp_id].push_back(symbolic_index.instantiate(ilp_prefetch_params));
            }
        }
    }

    // void CachePrefetchImpl::propagate_attributes_impl(const std::unordered_map<std::string, std::string> &attrs) {
    //     if (!attrs.count(OpAttribute::prefetch_predicate)) {
    //         LOG(FATAL) << "CachePrefetchImpl need prefetch_predicate attribute";
    //     }

    //     this->set_attribute(OpAttribute::prefetch_predicate, attrs.at(OpAttribute::prefetch_predicate));
    // }

    std::vector<std::shared_ptr<OpImplBase>>
    CachePrefetchImpl::get_available_implementations(std::shared_ptr<CUDAContext> cuda_context, InputSpec input_spec) {
        std::shared_ptr<OpImplBase> impl = std::make_shared<CachePrefetchImpl>(cuda_context, input_spec);

        return { impl };
    }
} // onefuser
} // core
} // op_impl