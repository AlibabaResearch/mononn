#include "mononn_engine/core/op/convolution.h"

#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace core {
namespace op {
using OpImpl = mononn_engine::core::op_impl::OpImplBase;
OpType Convolution::get_type() const { return OpType::convolution; }

std::vector<std::shared_ptr<OpImpl>>
Convolution::generate_candidate_implementation(
    std::shared_ptr<CUDAContext> context, Tier tier) const {
  LOG(FATAL) << "Not implemented";
}
}  // namespace op
}  // namespace core
}  // namespace mononn_engine