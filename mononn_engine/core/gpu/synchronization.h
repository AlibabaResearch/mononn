#pragma once
#include <string>

namespace mononn_engine {
namespace core {
namespace gpu {
class Synchronization {
 public:
  Synchronization() {}
  Synchronization(std::string _type, int _synchronization_level)
      : type(_type), synchronization_level(_synchronization_level) {}
  Synchronization(const char* _type, int _synchronization_level)
      : Synchronization(std::string(_type), _synchronization_level) {}

  static Synchronization const None;
  static Synchronization const Warp;
  static Synchronization const ThreadBlock;
  static Synchronization const Global;

  std::string to_string() const;

  bool operator<(const Synchronization& rhs) const;
  bool operator>(const Synchronization& rhs) const;
  bool operator!=(const Synchronization& rhs) const;
  bool operator==(const Synchronization& rhs) const;

  static std::string get_prerequisite_definition();

 private:
  std::string type;
  int synchronization_level;
};
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine