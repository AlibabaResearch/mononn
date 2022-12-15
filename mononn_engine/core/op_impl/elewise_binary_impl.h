#pragma once

#include <vector>
#include <string>
#include <memory>

#include "mononn_engine/core/tensor/tensor.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/op_impl/op_impl_base.h"
#include "mononn_engine/core/context/cuda_context.h"
#include "mononn_engine/core/tensor/math_op.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
    class ElewiseBinaryImpl : public OpImplBase {
    public:
        using Scalar = mononn_engine::core::tensor::Scalar;
        using Tensor = mononn_engine::core::tensor::Tensor;
        using FunctionInvocation = mononn_engine::core::semantic::FunctionInvocation;
        using OpType = mononn_engine::core::op::OpType;
        using CUDAContext = mononn_engine::core::context::CUDAContext;
        using MathOp = mononn_engine::core::tensor::MathOp;

        struct InputSpec {
            Tensor operand1;
            Tensor operand2;
            OpType op_type;
        };

        ElewiseBinaryImpl(std::shared_ptr<CUDAContext> _cuda_context, InputSpec _input_spec, Tensor _output)
        : cuda_context(_cuda_context), input_spec(_input_spec), output(_output) {}

        std::string generate_impl() const override;
        std::vector<Tensor> get_input_tensor() const override;
        std::vector<Tensor> get_output_tensor() const override;
        int get_elements_per_access() const override;
        void set_instruction_parallel_factor(int _ilp_factor) override;

        static std::vector<std::shared_ptr<OpImplBase>> get_available_implementations(
            std::shared_ptr<CUDAContext> cuda_context, 
            InputSpec input_spec, 
            Tensor output);

        static std::vector<std::shared_ptr<OpImplBase>> get_available_implementations_for_compare(
            std::shared_ptr<CUDAContext> cuda_context, 
            InputSpec input_spec, 
            Tensor output,
            MathOp math_op);

        static std::vector<std::shared_ptr<OpImplBase>> get_available_implementations_for_select(
                std::shared_ptr<CUDAContext> cuda_context,
                InputSpec input_spec,
                Tensor output,
                Tensor pred);

    private:
        InputSpec input_spec;
        int elements_per_access;
        std::shared_ptr<CUDAContext> cuda_context;
        Tensor output;
    };
}
}
}