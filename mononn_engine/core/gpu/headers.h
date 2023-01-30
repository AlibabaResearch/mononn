#pragma once

#include <string>

namespace mononn_engine {
namespace core {
namespace gpu {
class Headers {
 public:
  Headers(){};

  static std::string get_headers();
  static std::string get_headers_main_only();
  static std::string get_cuda_helpers();
  static std::string get_async_copy_headers();
  static std::string get_tuple_headers();
  static std::string get_tuple_shfl_headers();
};
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine