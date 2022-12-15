#include "mononn_engine/core/op/slice.h"
#include "tensorflow/core/platform/logging.h"
#include "mononn_engine/core/tensor/tensor.h"
#include "mononn_engine/core/tensor/tensor_spec.h"
#include "mononn_engine/core/op_impl/slice_impl.h"

namespace mononn_engine {
namespace core {
namespace op {
    using OpImpl = mononn_engine::core::op_impl::OpImplBase;
    using Tensor = mononn_engine::core::tensor::Tensor;
    using SliceImpl = mononn_engine::core::op_impl::SliceImpl;

    OpType Slice::get_type() const {
        return OpType::slice;
    }

    std::vector<std::shared_ptr<OpImpl>> Slice::generate_candidate_implementation(std::shared_ptr<CUDAContext> context, Tier tier) const {
        SliceImpl::InputSpec input_spec;
        input_spec.operand = this->get_operand(0)->get_output_tensor(0);
        input_spec.slice_starts = this->get_slice_starts();
        input_spec.slice_strides = this->get_slice_strides();
        input_spec.slice_limits = this->get_slice_limits();

        Tensor output = this->get_output_tensor(0);
        std::vector<std::shared_ptr<OpImpl>> impls = SliceImpl::get_available_implementations(context, input_spec, output);

        for (auto &impl : impls) {
            impl->set_hlo_text(this->get_hlo_text());
        }

        return impls;
    }

    void Slice::set_slice_starts(std::vector<int> _slice_starts) {
        this->slice_starts = _slice_starts;
    }

    std::vector<int> Slice::get_slice_starts() const {
        return this->slice_starts;
    }

    int Slice::get_slice_start(int index) const {
        if (index >= this->slice_starts.size()) {
            LOG(FATAL) << "Index out of range " << index << " but " << this->get_name() << " only have " << this->slice_starts.size();
        }

        return this->slice_starts[index];
    }

    void Slice::set_slice_limits(std::vector<int> _slice_limits) {

        this->slice_limits = _slice_limits;
    }

    std::vector<int> Slice::get_slice_limits() const {
        return this->slice_limits;
    }

    int Slice::get_slice_limit(int index) const {
        if (index >= this->slice_limits.size()) {
            LOG(FATAL) << "Index out of range " << index << " but " << this->get_name() << " only have " << this->slice_limits.size();
        }

        return this->slice_limits[index];
    }

    void Slice::set_slice_strides(std::vector<int> _slice_strides) {
        this->slice_strides = _slice_strides;
    }

    std::vector<int> Slice::get_slice_strides() const {
        return this->slice_strides;
    }

    int Slice::get_slice_stride(int index) const {
        if (index >= this->slice_strides.size()) {
            LOG(FATAL) << "Index out of range " << index << " but " << this->get_name() << " only have " << this->slice_strides.size();
        }

        return this->slice_strides[index];
    }
}
}
}