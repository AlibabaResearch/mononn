#include "mononn_engine/core/op/constant.h"
#include "tensorflow/core/platform/logging.h"
#include "mononn_engine/core/op_impl/constant_impl.h"
#include "mononn_engine/core/tensor/tensor.h"
#include "mononn_engine/core/tensor/tensor_spec.h"
#include "mononn_engine/core/tensor/dtype.h"

namespace mononn_engine {
namespace core {
namespace op {
    using OpImpl = mononn_engine::core::op_impl::OpImplBase;
    using ConstantImpl = mononn_engine::core::op_impl::ConstantImpl;
    using InputSpec = ConstantImpl::InputSpec;
    using Tensor = mononn_engine::core::tensor::Tensor;
    using TensorSpec = mononn_engine::core::tensor::TensorSpec;
    using Dtype = mononn_engine::core::tensor::Dtype;

    OpType Constant::get_type() const {
        return OpType::constant;
    }

    std::vector<std::shared_ptr<OpImpl>> Constant::generate_candidate_implementation(std::shared_ptr<CUDAContext> context, Tier tier) const {
        InputSpec input_spec;

        if (this->get_output_spec(0).element_count() == 1) {
            input_spec.value = this->get_value();
        }

        Tensor output(this->get_name(), this->get_output_spec(0));

        std::vector<std::shared_ptr<OpImpl>> impls = ConstantImpl::get_available_implementations(context, input_spec, output);

        for (auto &impl : impls) {
            impl->set_hlo_text(this->get_hlo_text());
        }

        return impls;
    }

    void Constant::set_value(std::string _value) {
        this->value = _value;
    }

    std::string Constant::get_value() const {
        return this->value;
    }

    bool Constant::is_scalar() const {
        return this->get_output_spec(0).element_count() == 1;
    }

    void Constant::set_data_float(std::vector<float> const &_data_float) {
        this->data_float = _data_float;
    }

    void Constant::set_data_half(std::vector<Eigen::half> const &_data_half) {
        this->data_half = _data_half;
    }

    const std::vector<float> &Constant::get_data_float() const {
        return this->data_float;
    }

    const std::vector<Eigen::half> &Constant::get_data_half() const {
        return this->data_half;
    }
}
}
}