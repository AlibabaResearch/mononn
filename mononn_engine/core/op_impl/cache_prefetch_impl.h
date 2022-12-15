#pragma once

#include "mononn_engine/core/op_impl/op_impl_base.h"
#include "mononn_engine/core/context/cuda_context.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
    using CUDAContext = mononn_engine::core::context::CUDAContext;

    class CachePrefetchImpl : public OpImplBase {
    public:
        using Tensor = mononn_engine::core::tensor::Tensor;
        using CUDAContext = mononn_engine::core::context::CUDAContext;

        struct InputSpec {
            Tensor operand;
        };

        CachePrefetchImpl(std::shared_ptr<CUDAContext> _cuda_context, InputSpec _input_spec)
        : cuda_context(_cuda_context), input_spec(_input_spec) {
            this->set_need_generate_with_index(true); // this is temporary hack.
        }

        std::string generate_with_index_impl() const override;
        int get_elements_per_access() const override;
        std::vector<Tensor> get_input_tensor() const override;
        std::vector<Tensor> get_output_tensor() const override;

        void set_instruction_parallel_factor(int _ilp_factor) override;

        static std::vector<std::shared_ptr<OpImplBase>> get_available_implementations(
                std::shared_ptr<CUDAContext> cuda_context,
                InputSpec input_spec);

    protected:
        void instantiate_concrete_index_impl(
                const std::vector<SymbolicIndexStamp> &symbolic_index_list,
                const std::map<std::string, std::string> &params,
                const std::string &loop_stride) override;
        void instantiate_ilp_concrete_index_impl(
                const std::vector<SymbolicIndexStamp> &symbolic_index_list,
                const std::map<std::string, std::string> &params,
                const std::string &loop_stride,
                const std::string &ilp_stride) override;

        // void propagate_attributes_impl(const std::unordered_map<std::string, std::string> &attrs) override;
    private:
        std::shared_ptr<CUDAContext> cuda_context;
        InputSpec input_spec;
    };

} // onefuser
} // core
} // op_impl


