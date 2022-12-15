#pragma once
#include <string>
#include <vector>
#include <memory>

#include "mononn_engine/core/tensor/tensor_spec.h"
#include "mononn_engine/core/op/op.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/tensor/scalar.h"
#include "mononn_engine/core/op_impl/reducer.h"
#include "mononn_engine/core/context/cuda_context.h"
#include "mononn_engine/codegen/reduction_functor_generator.h"

namespace mononn_engine {
namespace core {
namespace op {
    class Reduce : public Op {
    public:
        using Scalar = mononn_engine::core::tensor::Scalar;
        using TensorSpec = mononn_engine::core::tensor::TensorSpec;
        using Reducer = mononn_engine::core::op_impl::Reducer;
        using CUDAContext = mononn_engine::core::context::CUDAContext;
        using Tier = mononn_engine::core::op_annotation::LocalityTier::Tier;
        using ReductionFunctorGenerator = mononn_engine::codegen::ReductionFunctorGenerator;

        Reduce(std::string _name, std::vector<std::shared_ptr<Op>> _operands, std::vector<TensorSpec> _output_specs) 
        : Op(_name, _operands, _output_specs) {}

        OpType get_type() const override;
        std::vector<std::shared_ptr<OpImpl>> generate_candidate_implementation(std::shared_ptr<CUDAContext> context, Tier tier) const override;

        void set_dimension(int _dim);
        int get_dimension() const;

        void set_init_value(const Scalar &_init_value);
        const Scalar &get_init_value() const;

        // void set_reducer(Reducer _reducer);
        // Reducer get_reducer() const;

        bool need_pre_inner_loop_generation() const override;
        std::string generate_pre_inner_loop() const override;

        void set_reduction_functor_generator(const ReductionFunctorGenerator * _reduction_functor_generator);
    private:
        int dimension;
        Scalar init_value;
        // Reducer reducer;

        // Reduce node does not take ownership of generator as they may shared by multipled nodes.
        // The registry takes the ownership.
        const ReductionFunctorGenerator *reduction_functor_generator;
    };
}
}
}