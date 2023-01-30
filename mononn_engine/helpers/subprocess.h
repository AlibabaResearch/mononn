#pragma once
#include <string>

namespace mononn_engine {
namespace helpers {
class Subprocess {
 public:
  static std::string exec(std::string const cmd);

 private:
};
}  // namespace helpers
}  // namespace mononn_engine
