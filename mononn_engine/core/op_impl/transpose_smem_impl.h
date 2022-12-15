#pragma once
#include <memory>
#include <vector>
#include <string>

#include "mononn_engine/core/op_impl/op_impl_base.h"
#include "mononn_engine/core/tensor/tensor.h"
#include "mononn_engine/core/context/cuda_context.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
    using CUDAContext = mononn_engine::core::context::CUDAContext;
    using Tensor = mononn_engine::core::tensor::Tensor;

    class TransposeSmemImpl : public OpImplBase {
    public:

        struct InputSpec {
            Tensor operand;
            int batch_dim;
            int dim_r;
            int dim_c;
        };

        TransposeSmemImpl(std::shared_ptr<CUDAContext> _cuda_context, InputSpec _input_spec, Tensor _output)
                : cuda_context(_cuda_context), input_spec(_input_spec), output(_output) {}

        std::string generate_impl() const override;
        std::vector<Tensor> get_input_tensor() const override;
        std::vector<Tensor> get_output_tensor() const override;
        int get_elements_per_access() const override;

        void set_smem_tile_dim(int _dim);
        int get_smem_tile_dim() const;

        int get_smem_usage_in_bytes() const;
        void set_instruction_parallel_factor(int _ilp_factor) override;

        static std::vector<std::shared_ptr<OpImplBase>> get_available_implementations(
                std::shared_ptr<CUDAContext> cuda_context,
                InputSpec input_spec,
                Tensor output);

        static std::string get_prerequisite_definition();

    private:
        std::shared_ptr<CUDAContext> cuda_context;
        InputSpec input_spec;
        Tensor output;

        int smem_tile_dim;
    };
}
}
}