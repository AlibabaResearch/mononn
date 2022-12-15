#include "mononn_engine/core/gpu/cutlass/swizzle.h"

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
    Swizzle const Swizzle::GemmXThreadblockSwizzle = "cutlass::gemm::threadblock::GemmXThreadblockSwizzle";

    std::string Swizzle::to_string() const {
        return this->name;
    }
}
}
}
}
