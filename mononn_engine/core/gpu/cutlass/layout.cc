#include "mononn_engine/core/gpu/cutlass/layout.h"

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
Layout const Layout::RowMajor = "cutlass::layout::RowMajor";
Layout const Layout::ColumnMajor = "cutlass::layout::ColumnMajor";
Layout const Layout::TensorNHWC = "cutlass::layout::TensorNHWC";
Layout const Layout::TensorNCHW = "cutlass::layout::TensorNCHW";

std::string Layout::to_string() const { return this->name; }

bool Layout::operator==(Layout const& rhs) const {
  return this->name == rhs.name;
}

bool Layout::operator!=(Layout const& rhs) const { return !(*this == rhs); }
}  // namespace cutlass
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine