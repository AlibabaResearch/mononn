#include "mononn_engine/core/op/tuple.h"

#include "mononn_engine/core/tensor/tensor_spec.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace core {
namespace op {
using OpImpl = mononn_engine::core::op_impl::OpImplBase;
using TensorSpec = mononn_engine::core::tensor::TensorSpec;

OpType Tuple::get_type() const { return OpType::tuple; }

std::vector<std::shared_ptr<OpImpl>> Tuple::generate_candidate_implementation(
    std::shared_ptr<CUDAContext> context, Tier tier) const {
  LOG(FATAL) << "Not implemented";
}

bool Tuple::is_tuple() const { return true; }
}  // namespace op
}  // namespace core
}  // namespace mononn_engine