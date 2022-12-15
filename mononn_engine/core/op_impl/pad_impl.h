#pragma once
#include <vector>
#include <memory>
#include "mononn_engine/core/tensor/tensor.h"
#include "mononn_engine/core/op_impl/op_impl_base.h"
#include "mononn_engine/core/context/cuda_context.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
    class PadImpl : public OpImplBase {
    public:
        using Tensor = mononn_engine::core::tensor::Tensor;
        using CUDAContext = mononn_engine::core::context::CUDAContext;

        struct InputSpec {
            Tensor operand;
            std::vector<int> padding_low;
            std::vector<int> padding_high;
            std::string padding_value;
        };

        PadImpl(std::shared_ptr<CUDAContext> _cuda_context, InputSpec _input_spec, Tensor _output)
                : cuda_context(_cuda_context), input_spec(_input_spec), output(_output) {}

        std::string generate_impl() const override;
        std::string generate_with_index_impl() const override;

        int get_elements_per_access() const override;
        std::vector<Tensor> get_input_tensor() const override;
        std::vector<Tensor> get_output_tensor() const override;


//        std::string generate_if_statement_cond(std::vector<std::string> multi_index) const;
//        std::string generate_if_statement_begin(std::vector<std::string> multi_index) const;
//        std::string generate_if_statement_end() const;
//        std::string generate_else() const;
        void set_instruction_parallel_factor(int _ilp_factor) override;

        static std::vector<std::shared_ptr<OpImplBase>> get_available_implementations(
                std::shared_ptr<CUDAContext> cuda_context,
                InputSpec input_spec,
                Tensor output);
    private:
        std::shared_ptr<CUDAContext> cuda_context;
        InputSpec input_spec;
        Tensor output;
    };
}
}
}
