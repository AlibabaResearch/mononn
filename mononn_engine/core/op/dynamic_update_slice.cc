#include "mononn_engine/core/op/dynamic_update_slice.h"
#include "mononn_engine/core/op_impl/dynamic_update_slice_impl.h"

namespace mononn_engine {
namespace core {
namespace op {
    using DynamicUpdateSliceImpl = mononn_engine::core::op_impl::DynamicUpdateSliceImpl;
    using OpImplBase = mononn_engine::core::op_impl::OpImplBase;

    OpType DynamicUpdateSlice::get_type() const {
        return OpType::dynamic_update_slice;
    }

    std::vector<std::shared_ptr<OpImplBase>> DynamicUpdateSlice::generate_candidate_implementation(std::shared_ptr<CUDAContext> context, Tier tier) const {
        DynamicUpdateSliceImpl::InputSpec input_spec;
        
        for (auto const &operand : this->operands) {
            input_spec.operands.push_back(operand->get_output_tensor(0));
        }

        return DynamicUpdateSliceImpl::get_available_implementations(
            context, input_spec, this->get_output_tensor(0));
    }
}
}
}