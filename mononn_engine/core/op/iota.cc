#include "mononn_engine/core/op/iota.h"
#include "tensorflow/core/platform/logging.h"
#include "mononn_engine/core/tensor/tensor_spec.h"
#include "mononn_engine/core/op_impl/iota_impl.h"

namespace mononn_engine {
namespace core {
namespace op {
    using OpImpl = mononn_engine::core::op_impl::OpImplBase;
    using Tensor = mononn_engine::core::tensor::Tensor;
    using IotaImpl = mononn_engine::core::op_impl::IotaImpl;

    OpType Iota::get_type() const {
        return OpType::iota;
    }

    std::vector<std::shared_ptr<OpImpl>> Iota::generate_candidate_implementation(std::shared_ptr<CUDAContext> context, Tier tier) const {
        IotaImpl::InputSpec input_spec;
        input_spec.iota_dimension = this->get_iota_dimension();

        Tensor output = this->get_output_tensor(0);
        std::vector<std::shared_ptr<OpImpl>> impls = IotaImpl::get_available_implementations(context, input_spec, output);

        for (auto &impl : impls) {
            impl->set_hlo_text(this->get_hlo_text());
        }

        return impls;
    }

    void Iota::set_iota_dimension(int _iota_dimension) {
        this->iota_dimension = _iota_dimension;
    }

    int Iota::get_iota_dimension() const {
        return this->iota_dimension;
    }
}
}
}