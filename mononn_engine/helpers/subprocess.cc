#include "mononn_engine/helpers/subprocess.h"

#include <array>
#include <cstdio>

#include "mononn_engine/helpers/macros.h"

namespace mononn_engine {
namespace helpers {

std::string Subprocess::exec(std::string const cmd) {
  std::array<char, 128> buffer;
  std::string result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"),
                                                pclose);
  if (!pipe) {
    LOG(FATAL) << "popen() failed!";
  }

  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }

  return result;
}
}  // namespace helpers
}  // namespace mononn_engine