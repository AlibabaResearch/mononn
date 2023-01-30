#include "mononn_engine/core/op/transpose.h"

#include "mononn_engine/core/op_impl/transpose_impl.h"
#include "mononn_engine/core/tensor/tensor_spec.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace core {
namespace op {
using OpImpl = mononn_engine::core::op_impl::OpImplBase;
using TransposeImpl = mononn_engine::core::op_impl::TransposeImpl;
using Tensor = mononn_engine::core::tensor::Tensor;

void Transpose::set_permute(std::vector<int> _permute) {
  this->permute = _permute;
}

std::vector<int> Transpose::get_permute() const { return this->permute; }

bool Transpose::need_smem() const {
  return this->permute.back() != (int)this->permute.size() - 1;
}

OpType Transpose::get_type() const { return OpType::transpose; }

std::vector<std::shared_ptr<OpImpl>>
Transpose::generate_candidate_implementation(
    std::shared_ptr<CUDAContext> context, Tier tier) const {
  TransposeImpl::InputSpec input_spec;
  input_spec.operand = this->get_operand(0)->get_output_tensor(0);
  input_spec.perm = this->get_permute();
  Tensor output = this->get_output_tensor(0);

  std::vector<std::shared_ptr<OpImpl>> impls =
      TransposeImpl::get_available_implementations(context, input_spec, output);

  for (auto& impl : impls) {
    impl->set_hlo_text(this->get_hlo_text());
  }

  return impls;
}
}  // namespace op
}  // namespace core
}  // namespace mononn_engine