#include "mononn_engine/core/op_impl/output_impl_base.h"
#include "mononn_engine/helpers/helpers.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
    bool OutputImplBase::need_pre_inner_loop_generation() const {
        return false;
    }

    std::string OutputImplBase::generate_pre_inner_loop() const {
        LOG(FATAL) << "Unimplemented";
    }
}
}
}