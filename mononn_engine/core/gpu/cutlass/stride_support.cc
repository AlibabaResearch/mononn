#include "mononn_engine/core/gpu/cutlass/stride_support.h"

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
const StrideSupport StrideSupport::kStrided =
    "cutlass::conv::StrideSupport::kStrided";
const StrideSupport StrideSupport::kUnity =
    "cutlass::conv::StrideSupport::kUnity";

std::string StrideSupport::to_string() const { return this->name; }
}  // namespace cutlass
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine