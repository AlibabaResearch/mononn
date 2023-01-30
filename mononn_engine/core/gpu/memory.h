#pragma once

#include <string>

#include "mononn_engine/core/tensor/dtype.h"

namespace mononn_engine {
namespace core {
namespace gpu {
class Memory {
 public:
  using Dtype = mononn_engine::core::tensor::Dtype;

  enum AccessFlavor {
    // public:
    // AccessFlavor(std::string _flavor) : flavor(_flavor) {}
    // AccessFlavor(const char * _flavor) : AccessFlavor(std::string(_flavor))
    // {}

    // enum Type {
    REGULAR = 0,
    EXPLICIT_PTX,
    EXPLICIT_PTX_EVICT_LAST,
    // do not perform reinterpret_cast on buffer pointer. Assume they natually
    // have the same type.
    // only support 1d array at this moment.
    STRONG_TYPED
    // };
    // static AccessFlavor const REGULAR;
    // static AccessFlavor const EXPLICIT_PTX;
    // static AccessFlavor const EXPLICIT_PTX_EVICT_LAST;

    // static AccessFlavor const STRONG_TYPED;

    // static std::string to_string(AccessFlavor::Type flavor);
    //     bool operator == (AccessFlavor const &rhs) const;

    // private:
    //     std::string flavor;
  };

  static std::string AccessFlavorToString(AccessFlavor access_flavor);

  static std::string read(AccessFlavor access_flavor, Dtype access_type,
                          std::string var_name, std::string src_ptr,
                          std::string offset, bool define_variable,
                          std::string pred = "true",
                          std::string default_value = "0");
  static std::string write(AccessFlavor access_flavor, Dtype access_type,
                           std::string var_name, std::string dst_ptr,
                           std::string offset, std::string pred = "true");
  static std::string broadcast_read(Dtype access_type, std::string var_name,
                                    std::string src_ptr, std::string offset,
                                    bool define_variable,
                                    std::string pred = "true",
                                    std::string default_value = "0");
  static std::string prefetch_l1(Dtype access_type, std::string ptr,
                                 std::string offset);
  static std::string prefetch_l1(Dtype access_type, std::string ptr,
                                 std::string offset, std::string predicate);
  static std::string prefetch_l2(Dtype access_type, std::string ptr,
                                 std::string offset);
  static std::string prefetch_l2(Dtype access_type, std::string ptr,
                                 std::string offset, std::string predicate);

 private:
  static std::string read_regular(Dtype access_type, std::string var_name,
                                  std::string src_ptr, std::string offset,
                                  bool define_variable, std::string pred,
                                  std::string default_value);
  static std::string read_explicit_ptx(Dtype access_type, std::string var_name,
                                       std::string src_ptr, std::string offset,
                                       bool define_variable, std::string pred,
                                       std::string default_value);
  static std::string read_explicit_ptx_evict_last(
      Dtype access_type, std::string var_name, std::string src_ptr,
      std::string offset, bool define_variable, std::string pred,
      std::string default_value);

  static std::string write_regular(Dtype access_type, std::string var_name,
                                   std::string dst_ptr, std::string offset,
                                   std::string pred);
  static std::string write_explicit_ptx(Dtype access_type, std::string var_name,
                                        std::string dst_ptr, std::string offset,
                                        std::string pred);

  static std::string read_regular_strong_typed_1d_array(
      Dtype access_type, std::string var_name, std::string src_ptr,
      std::string offset, bool define_variable, std::string pred,
      std::string default_value);
  static std::string write_regular_strong_typed_1d_array(Dtype access_type,
                                                         std::string var_name,
                                                         std::string dst_ptr,
                                                         std::string offset,
                                                         std::string pred);
};
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine