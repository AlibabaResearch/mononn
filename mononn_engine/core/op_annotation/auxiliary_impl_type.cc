#include "mononn_engine/core/op_annotation/auxiliary_impl_type.h"

namespace mononn_engine {
namespace core {
namespace op_annotation {
const std::string AuxiliaryImplType::buffer_in_register = "buffer_in_register";
const std::string AuxiliaryImplType::explicit_output_node =
    "explicit_output_node";
const std::string AuxiliaryImplType::cache_prefetch = "cache_prefetch";
}  // namespace op_annotation
}  // namespace core
}  // namespace mononn_engine