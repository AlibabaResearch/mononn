#include "mononn_engine/core/op_impl/tuple_impl.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
    std::string TupleImpl::generate_with_index_impl() const {
        LOG(FATAL) << "";
    }

    std::vector<Tensor> TupleImpl::get_input_tensor() const {
        return std::vector<Tensor>();
    }

    std::vector<Tensor> TupleImpl::get_output_tensor() const {
        return std::vector<Tensor>();
    }

    int TupleImpl::get_elements_per_access() const {
        return 0;
    }
    
    void TupleImpl::set_instruction_parallel_factor(int _ilp_factor) {
        LOG(FATAL) << "Unimplemented";
    }

    std::vector<std::shared_ptr<OpImplBase>>
    TupleImpl::get_available_implementations(std::shared_ptr<mononn_engine::core::context::CUDAContext> cuda_context,
                                             TupleImpl::InputSpec input_spec, TupleImpl::Tensor output) {
        return std::vector<std::shared_ptr<OpImplBase>>();
    }
}
}
}