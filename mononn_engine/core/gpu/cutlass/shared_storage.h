#pragma once

#include <string>

#include "mononn_engine/core/gpu/cutlass/gemm_shape.h"

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
class SharedStorage {
 public:
  static int get_shared_storage_size(GemmShape ThreadblockShape, int stages,
                                     int A_sizeof_bytes, int B_sizeof_bytes);

 private:
};
}  // namespace cutlass
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine