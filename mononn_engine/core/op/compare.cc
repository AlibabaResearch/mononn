#include "mononn_engine/core/op/compare.h"
#include "tensorflow/core/platform/logging.h"
#include "mononn_engine/core/tensor/tensor.h"
#include "mononn_engine/core/tensor/tensor_spec.h"
#include "mononn_engine/core/op_impl/elewise_binary_impl.h"

namespace mononn_engine {
namespace core {
namespace op {
    using OpImpl = mononn_engine::core::op_impl::OpImplBase;
    using MathOp = mononn_engine::core::tensor::MathOp;
    using ElewiseBinaryImpl = mononn_engine::core::op_impl::ElewiseBinaryImpl;
    using Tensor = mononn_engine::core::tensor::Tensor;

    OpType Compare::get_type() const {
        return OpType::compare;
    }

    std::vector<std::shared_ptr<OpImpl>> Compare::generate_candidate_implementation(std::shared_ptr<CUDAContext> context, Tier tier) const {
        ElewiseBinaryImpl::InputSpec input_spec;
        input_spec.op_type = this->get_type();
        input_spec.operand1 = this->get_operand(0)->get_output_tensor(0);
        input_spec.operand2 = this->get_operand(1)->get_output_tensor(0);
        Tensor output = this->get_output_tensor(0);
        std::vector<std::shared_ptr<OpImpl>> impls = ElewiseBinaryImpl::get_available_implementations_for_compare(context, input_spec, output, this->get_comparator());

        for (auto &impl : impls) {
            impl->set_hlo_text(this->get_hlo_text());
        }

        return impls;
    }

    void Compare::set_comparator(MathOp _comparator) {
        this->comparator = _comparator;
    }

    MathOp Compare::get_comparator() const {
        return this->comparator;
    }
}
}
}