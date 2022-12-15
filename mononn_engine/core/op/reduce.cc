#include "mononn_engine/core/op/reduce.h"
#include "tensorflow/core/platform/logging.h"
#include "mononn_engine/core/op_annotation/locality_tier.h"
#include "mononn_engine/core/tensor/tensor.h"
#include "mononn_engine/core/op_impl/reduce_impl.h"

namespace mononn_engine {
namespace core {
namespace op {
    using Scalar = mononn_engine::core::tensor::Scalar;
    using Reducer = mononn_engine::core::op_impl::Reducer;
    using OpImpl = mononn_engine::core::op_impl::OpImplBase;
    using CUDAContext = mononn_engine::core::context::CUDAContext;
    using Tier = mononn_engine::core::op_annotation::LocalityTier::Tier;
    using Tensor = mononn_engine::core::tensor::Tensor;
    using ReduceImpl = mononn_engine::core::op_impl::ReduceImpl;
    using Dtype = mononn_engine::core::tensor::Dtype;
    using Scalar = mononn_engine::core::tensor::Scalar;
    
    OpType Reduce::get_type() const {
        return OpType::reduce;
    }

    std::vector<std::shared_ptr<OpImpl>> Reduce::generate_candidate_implementation(std::shared_ptr<CUDAContext> context, Tier tier) const {
        ReduceImpl::InputSpec input_spec;
        // std::shared_ptr<Op> operand = this->get_operand(0);

        // input_spec.operand = Tensor(operand->get_name(), operand->get_output_spec(0));
        // input_spec.init_value = this->get_init_value().get_value();
        // input_spec.reducer = this->get_reducer();

        for (int operand_id = 0; operand_id < this->get_operand_count() / 2; ++operand_id) {
            input_spec.operands.emplace_back(this->get_operand(operand_id)->get_name(), this->get_operand(operand_id)->get_output_spec(0));
        }

        input_spec.init_value = this->get_init_value();
        input_spec.dimension = this->get_dimension();
        input_spec.tier = tier;
        input_spec.reduction_functor_generator = this->reduction_functor_generator;
        

        Tensor output(this->get_name(), this->get_output_spec(0));

        std::vector<Dtype> init_type_list;
        std::vector<std::string> init_value_list;

        for (int operand_idx = this->get_operand_count() / 2; operand_idx < this->get_operand_count(); ++operand_idx) {
            init_type_list.push_back(this->get_operand(operand_idx)->get_output_spec(0).get_dtype().get_primitive_type());
            // init_value_list.push_back(input_spec.operands[param_idx].get_name());
            init_value_list.push_back(input_spec.init_value.get_values_in_list()[operand_idx - this->get_operand_count() / 2]);
        }

        input_spec.reduce_accum = Scalar(output.get_name() + "_accum", init_type_list, init_value_list);
        
        std::vector<std::shared_ptr<OpImpl>> impls = ReduceImpl::get_available_implementations(context, input_spec, output);

        for (auto &impl : impls) {
            impl->set_hlo_text(this->get_hlo_text());
        }

        return impls;
    }

    void Reduce::set_dimension(int _dim) {
        if (_dim != this->operands[0]->get_output_spec(0).rank() - 1) {
            LOG(FATAL) << "Unsupported reduction in operator: " << this->get_name()
                << ". Reduction dimension " << _dim << " reduction operand " << this->operands[0]->get_name() << " shape " << this->operands[0]->get_output_spec(0).to_string();
        }

        this->dimension = _dim;
    }

    int Reduce::get_dimension() const {
        return this->dimension;
    }

    void Reduce::set_init_value(const Scalar &_init_value) {
        this->init_value = _init_value;
    }

    const Scalar &Reduce::get_init_value() const {
        return this->init_value;
    }

    // void Reduce::set_reducer(Reducer _reducer) {
    //     this->reducer = _reducer;
    // }

    // Reducer Reduce::get_reducer() const {
    //     return this->reducer;
    // }

    bool Reduce::need_pre_inner_loop_generation() const {
        return true;
    }

    std::string Reduce::generate_pre_inner_loop() const {
        Scalar reduce_accum = this->get_implementation()->as<ReduceImpl>()->get_reduce_accum();
        return reduce_accum.get_definition_with_value() + "\n";
    }

    void Reduce::set_reduction_functor_generator(const ReductionFunctorGenerator * _reduction_functor_generator) {
        this->reduction_functor_generator = _reduction_functor_generator;
    }
}
}
}