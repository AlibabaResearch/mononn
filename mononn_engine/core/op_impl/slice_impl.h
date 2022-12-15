#pragma once
#include <vector>
#include <memory>
#include "mononn_engine/core/tensor/tensor.h"
#include "mononn_engine/core/op_impl/op_impl_base.h"
#include "mononn_engine/core/context/cuda_context.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
    class SliceImpl : public OpImplBase {
    public:
        using Tensor = mononn_engine::core::tensor::Tensor;
        using CUDAContext = mononn_engine::core::context::CUDAContext;

        struct InputSpec {
            Tensor operand;
            std::vector<int> slice_starts;
            std::vector<int> slice_limits;
            std::vector<int> slice_strides;
        };

        SliceImpl(std::shared_ptr<CUDAContext> _cuda_context, InputSpec _input_spec, Tensor _output)
                : cuda_context(_cuda_context), input_spec(_input_spec), output(_output) {}

        std::string generate_impl() const override;
        int get_elements_per_access() const override;
        std::vector<Tensor> get_input_tensor() const override;
        std::vector<Tensor> get_output_tensor() const override;
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
