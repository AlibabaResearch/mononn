#include "mononn_engine/core/op_impl/global_sync_impl.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
    using Tensor = GlobalSyncImpl::Tensor;
    using TensorSpec = mononn_engine::core::tensor::TensorSpec;

    std::string GlobalSyncImpl::generate_impl() const {
        return "synchronization::grid_sync();";
    }

    std::vector<Tensor> GlobalSyncImpl::get_input_tensor() const {
        return {};
    }

    std::vector<Tensor> GlobalSyncImpl::get_output_tensor() const {
        return {};
    }

    int GlobalSyncImpl::get_elements_per_access() const {
        return -1;
    }

    void GlobalSyncImpl::set_instruction_parallel_factor(int _ilp_factor) {
        
        LOG(FATAL) << "Unimplemented";
    }

    std::vector<std::shared_ptr<OpImplBase>>
    GlobalSyncImpl::get_available_implementations(std::shared_ptr<CUDAContext> cuda_context) {
        std::shared_ptr<GlobalSyncImpl> impl = std::make_shared<GlobalSyncImpl>(cuda_context);

        return { std::static_pointer_cast<OpImplBase>(impl) };
    }
}
}
}