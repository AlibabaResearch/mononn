#include "mononn_engine/core/gpu/cutlass/shared_storage.h"

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
int SharedStorage::get_shared_storage_size(GemmShape ThreadblockShape,
                                           int stages, int A_sizeof_bytes,
                                           int B_sizeof_bytes) {
  return (ThreadblockShape.mk() * A_sizeof_bytes +
          ThreadblockShape.nk() * B_sizeof_bytes) *
         stages;
}
}  // namespace cutlass
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine