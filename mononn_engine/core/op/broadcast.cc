#include "mononn_engine/core/op/broadcast.h"
#include "tensorflow/core/platform/logging.h"
#include "mononn_engine/core/tensor/tensor.h"
#include "mononn_engine/core/tensor/tensor_spec.h"
#include "mononn_engine/core/op_impl/broadcast_impl.h"

namespace mononn_engine {
namespace core {
namespace op {
    using OpImpl = mononn_engine::core::op_impl::OpImplBase;
    using BroadcastImpl = mononn_engine::core::op_impl::BroadcastImpl;

    OpType Broadcast::get_type() const {
        return OpType::broadcast;
    }

    std::vector<std::shared_ptr<OpImpl>> Broadcast::generate_candidate_implementation(std::shared_ptr<CUDAContext> context, Tier tier) const {
        BroadcastImpl::InputSpec input_spec;
        input_spec.operand = this->get_operand(0)->get_output_tensor(0);
        input_spec.dimensions = this->get_dimensions();
        Tensor output = this->get_output_tensor(0);

        std::vector<std::shared_ptr<OpImpl>> impls = BroadcastImpl::get_available_implementations(context, input_spec, output);

        for (auto &impl : impls) {
            impl->set_hlo_text(this->get_hlo_text());
        }

        return impls;
    }

    void Broadcast::set_dimensions(std::vector<int> _dimensions) {
        this->dimensions = _dimensions;
    }

    std::vector<int> Broadcast::get_dimensions() const {
        return this->dimensions;
    }
}
}
}