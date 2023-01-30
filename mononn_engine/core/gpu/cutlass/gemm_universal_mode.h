#pragma once

#include <string>

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
class GemmUniversalMode {
 public:
  GemmUniversalMode() {}
  GemmUniversalMode(std::string _mode) : mode(_mode) {}
  GemmUniversalMode(const char* _mode)
      : GemmUniversalMode(std::string(_mode)) {}

  static GemmUniversalMode const kGemm;
  static GemmUniversalMode const kBatched;
  static GemmUniversalMode const kArray;

  std::string to_string() const;

 private:
  std::string mode;
};
}  // namespace cutlass
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine