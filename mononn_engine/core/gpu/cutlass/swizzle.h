#pragma once
#include <string>

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
class Swizzle {
 public:
  Swizzle(std::string _name) : name(_name) {}
  Swizzle(const char* _name) : Swizzle(std::string(_name)) {}

  static Swizzle const GemmXThreadblockSwizzle;

  std::string to_string() const;

 private:
  std::string name;
};
}  // namespace cutlass
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine