#include "mononn_engine/core/op/sign.h"
#include "tensorflow/core/platform/logging.h"
#include "mononn_engine/core/tensor/tensor_spec.h"
#include "mononn_engine/core/op_impl/elewise_unary_impl.h"

namespace mononn_engine {
namespace core {
namespace op {
    using OpImpl = mononn_engine::core::op_impl::OpImplBase;
    using ElewiseUnaryImpl = mononn_engine::core::op_impl::ElewiseUnaryImpl;

    OpType Sign::get_type() const {
        return OpType::sign;
    }

    std::vector<std::shared_ptr<OpImpl>> Sign::generate_candidate_implementation(std::shared_ptr<CUDAContext> context, Tier tier) const {
        ElewiseUnaryImpl::InputSpec input_spec;
        input_spec.operand = this->get_operand(0)->get_output_tensor(0);
        input_spec.op_type = this->get_type();

        Tensor output = this->get_output_tensor(0);

        std::vector<std::shared_ptr<OpImpl>> impls = ElewiseUnaryImpl::get_available_implementations(context, input_spec, output);

        for (auto &impl : impls) {
            impl->set_hlo_text(this->get_hlo_text());
        }

        return impls;
    }
}
}
}