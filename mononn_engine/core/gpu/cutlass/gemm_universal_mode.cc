#include "mononn_engine/core/gpu/cutlass/gemm_universal_mode.h"

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
    GemmUniversalMode const GemmUniversalMode::kGemm = "cutlass::gemm::GemmUniversalMode::kGemm";
    GemmUniversalMode const GemmUniversalMode::kBatched = "cutlass::gemm::GemmUniversalMode::kBatched";
    GemmUniversalMode const GemmUniversalMode::kArray = "cutlass::gemm::GemmUniversalMode::kArray";

    std::string GemmUniversalMode::to_string() const {
        return this->mode;
    }
}
}
}
}