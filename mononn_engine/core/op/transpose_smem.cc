#include "mononn_engine/core/op/transpose_smem.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/op_impl//transpose_smem_impl.h"
#include "mononn_engine/core/tensor/tensor.h"
#include "mononn_engine/core/op_impl/op_impl_base.h"

namespace mononn_engine {
namespace core {
namespace op {
    using TransposeSmemImpl = mononn_engine::core::op_impl::TransposeSmemImpl;
    using Tensor = mononn_engine::core::tensor::Tensor;
    using OpImplBase = mononn_engine::core::op_impl::OpImplBase;

    OpType TransposeSmem::get_type() const {
        return OpType::transpose_smem;
    }

    std::vector<std::shared_ptr<OpImplBase>>
    TransposeSmem::generate_candidate_implementation(std::shared_ptr<CUDAContext> context,
                                                     TransposeSmem::Tier tier) const {
        TransposeSmemImpl::InputSpec input_spec;

        input_spec.operand = Tensor(this->get_operand(0)->get_name(), this->get_operand(0)->get_output_spec(0));
        input_spec.batch_dim = this->get_batch_dim();
        input_spec.dim_r = this->get_dim_r();
        input_spec.dim_c = this->get_dim_c();

        Tensor output(this->get_name(), this->get_output_spec(0));

        return TransposeSmemImpl::get_available_implementations(context, input_spec, output);
    }

    void TransposeSmem::set_batch_dim(int _batch_dim) {
        this->batch_dim = _batch_dim;
    }

    int TransposeSmem::get_batch_dim() const {
        return this->batch_dim;
    }

    void TransposeSmem::set_dim_r(int _dim_r) {
        this->dim_r = _dim_r;
    }

    int TransposeSmem::get_dim_r() const {
        return this->dim_r;
    }

    void TransposeSmem::set_dim_c(int _dim_c) {
        this->dim_c = _dim_c;
    }

    int TransposeSmem::get_dim_c() const {
        return this->dim_c;
    }
}
}
}