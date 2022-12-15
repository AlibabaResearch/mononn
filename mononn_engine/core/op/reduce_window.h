#pragma once
#include <string>
#include <vector>
#include <memory>

#include "mononn_engine/core/tensor/tensor_spec.h"
#include "mononn_engine/core/op/op.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/tensor/scalar.h"
#include "mononn_engine/codegen/reduction_functor_generator.h"

namespace mononn_engine {
namespace core {
namespace op {
    class ReduceWindow : public Op {
    public:
        using TensorSpec = mononn_engine::core::tensor::TensorSpec;
        using Tier = mononn_engine::core::op_annotation::LocalityTier::Tier;
        using Scalar = mononn_engine::core::tensor::Scalar;
        using ReductionFunctorGenerator = mononn_engine::codegen::ReductionFunctorGenerator;
        
        ReduceWindow(std::string _name, std::vector<std::shared_ptr<Op>> _operands, std::vector<TensorSpec> _output_specs) 
        : Op(_name, _operands, _output_specs) {}
        OpType get_type() const override;
        std::vector<std::shared_ptr<OpImpl>> generate_candidate_implementation(std::shared_ptr<CUDAContext> context, Tier tier) const override;

        void set_init_value(const Scalar &_init_value);
        const Scalar &get_init_value() const;

        void set_filter_size(const std::vector<int> &_filter_size);
        const std::vector<int>& get_filter_size() const;

        void set_filter_stride(const std::vector<int> &_filter_stride);
        const std::vector<int>& get_filter_stride() const;

        void set_padding_low(const std::vector<int> &_padding_low);
        const std::vector<int>& get_padding_low() const;

        void set_padding_high(const std::vector<int> &_padding_high);
        const std::vector<int>& get_padding_high() const;

        void set_reduction_functor_generator(const ReductionFunctorGenerator * _reduction_functor_generator);
        const ReductionFunctorGenerator *get_reduction_functor_generator() const;

        std::vector<std::vector<int>> get_window_positions() const;

    private:

        std::vector<int> filter_size;
        std::vector<int> filter_stride;
        std::vector<int> padding_low;
        std::vector<int> padding_high;

        Scalar init_value;

        const ReductionFunctorGenerator *reduction_functor_generator;
    };
}
}
}