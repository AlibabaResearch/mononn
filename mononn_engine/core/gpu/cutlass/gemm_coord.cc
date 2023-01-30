#include "mononn_engine/core/gpu/cutlass/gemm_coord.h"

#include "mononn_engine/helpers/string_helpers.h"
namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
std::string GemmCoord::to_string() const {
  return mononn_engine::helpers::string_format(
      "cutlass::gemm::GemmCoord(%d, %d, %d)", this->m, this->n, this->k);
}
}  // namespace cutlass
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine