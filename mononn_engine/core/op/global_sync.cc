#include "mononn_engine/core/op/global_sync.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/op_impl/op_impl_base.h"
#include "mononn_engine/core/op_impl/global_sync_impl.h"

namespace mononn_engine {
namespace core {
namespace op {
    using OpImpl = GlobalSync::OpImpl;
    using GlobalSyncImpl = mononn_engine::core::op_impl::GlobalSyncImpl;

    OpType GlobalSync::get_type() const {
        return OpType::global_sync;
    }

    std::vector<std::shared_ptr<OpImpl>> GlobalSync::generate_candidate_implementation(std::shared_ptr<CUDAContext> context, Tier tier) const {
        auto impls = GlobalSyncImpl::get_available_implementations(context);

        for (auto &impl : impls) {
            impl->set_hlo_text(this->get_hlo_text());
        }

        return impls;
    }
}
}
}