#include "mononn_engine/core/gpu/cutlass/iterator_algorithm.h"

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
const IteratorAlgorithm IteratorAlgorithm::kOptimized =
    "cutlass::conv::IteratorAlgorithm::kOptimized";
const IteratorAlgorithm IteratorAlgorithm::kAnalytic =
    "cutlass::conv::IteratorAlgorithm::kAnalytic";

std::string IteratorAlgorithm::to_string() const { return this->name; }
}  // namespace cutlass
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine