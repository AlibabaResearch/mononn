#include "mononn_engine/core/op/gather.h"
#include "tensorflow/core/platform/logging.h"
#include "mononn_engine/core/tensor/tensor_spec.h"
#include "mononn_engine/core/op_impl/gather_impl.h"

namespace mononn_engine {
namespace core {
namespace op {
    using OpImpl = mononn_engine::core::op_impl::OpImplBase;
    using Tensor = mononn_engine::core::tensor::Tensor;
    using GatherImpl = mononn_engine::core::op_impl::GatherImpl;

    OpType Gather::get_type() const {
        return OpType::gather;
    }

    std::vector<std::shared_ptr<OpImpl>> Gather::generate_candidate_implementation(std::shared_ptr<CUDAContext> context, Tier tier) const {
        GatherImpl::InputSpec input_spec;
        input_spec.operand = this->get_operand(0)->get_output_tensor(0);
        input_spec.start_indices = this->get_operand(1)->get_output_tensor(0);
        input_spec.index_vector_dim = this->get_index_vector_dim();
        input_spec.offset_dims = this->get_offset_dims();
        input_spec.slice_sizes = this->get_slice_sizes();
        input_spec.collapsed_slice_dims = this->get_collapsed_slice_dims();
        input_spec.start_index_map = this->get_start_index_map();
        input_spec.indices_are_sorted = this->get_indices_are_sorted();
        input_spec.unique_indices = this->get_unique_indices();

        Tensor output = this->get_output_tensor(0);

        std::vector<std::shared_ptr<OpImpl>> impls = GatherImpl::get_available_implementations(context, input_spec, output);

        for (auto &impl : impls) {
            impl->set_hlo_text(this->get_hlo_text());
        }

        return impls;
    }

    void Gather::set_offset_dims(std::vector<int> _offset_dims) {
        this->offset_dims = _offset_dims;
    }

    std::vector<int> Gather::get_offset_dims() const {
        return this->offset_dims;
    }

    void Gather::set_collapsed_slice_dims(std::vector<int> _collapsed_slice_dims) {
        this->collapsed_slice_dims = _collapsed_slice_dims;
    }

    std::vector<int> Gather::get_collapsed_slice_dims() const {
        return this->collapsed_slice_dims;
    }

    void Gather::set_start_index_map(std::vector<int> _start_index_map) {
        this->start_index_map = _start_index_map;
    }

    std::vector<int> Gather::get_start_index_map() const {
        return this->start_index_map;
    }

    void Gather::set_index_vector_dim(int _index_vector_dim) {
        this->index_vector_dim = _index_vector_dim;
    }

    int Gather::get_index_vector_dim() const {
        return this->index_vector_dim;
    }

    void Gather::set_slice_sizes(std::vector<int> _slice_sizes) {
        this->slice_sizes = _slice_sizes;
    }

    std::vector<int> Gather::get_slice_sizes() const {
        return this->slice_sizes;
    }

    void Gather::set_indices_are_sorted(bool _indices_are_sorted) {
        this->indices_are_sorted = _indices_are_sorted;
    }

    bool Gather::get_indices_are_sorted() const {
        return this->indices_are_sorted;
    }

    void Gather::set_unique_indices(bool _unique_indices) {
        this->unique_indices = _unique_indices;
    }

    bool Gather::get_unique_indices() const {
        return this->unique_indices;
    }
}
}
}