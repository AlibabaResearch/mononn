// Copyright 2023 The MonoNN Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mononn_engine/core/gpu/memory.h"

#include "mononn_engine/helpers/string_helpers.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace core {
namespace gpu {
// Memory::AccessFlavor const Memory::AccessFlavor::REGULAR = "REGUMAR";
// Memory::AccessFlavor const Memory::AccessFlavor::EXPLICIT_PTX =
// "EXPLICIT_PTX"; Memory::AccessFlavor const
// Memory::AccessFlavor::EXPLICIT_PTX_EVICT_LAST = "EXPLICIT_PTX_EVICT_LAST";
// Memory::AccessFlavor const Memory::AccessFlavor::STRONG_TYPED =
// "STRONG_TYPED";

// std::string Memory::AccessFlavor::to_string(AccessFlavor::Type flavor) {

// }

// bool Memory::AccessFlavor::operator== (Memory::AccessFlavor const &rhs) const
// {
//     return this->flavor == rhs.flavor;
// }

std::string Memory::AccessFlavorToString(AccessFlavor access_flavor) {
  if (access_flavor == AccessFlavor::REGULAR) {
    return "REGULAR";
  }

  if (access_flavor == AccessFlavor::EXPLICIT_PTX) {
    return "EXPLICIT_PTX";
  }

  if (access_flavor == AccessFlavor::EXPLICIT_PTX_EVICT_LAST) {
    return "EXPLICIT_PTX_EVICT_LAST";
  }

  if (access_flavor == AccessFlavor::STRONG_TYPED) {
    return "STRONG_TYPED";
  }

  LOG(FATAL) << "Unexpected access flavor " << (int)access_flavor;
}

std::string Memory::read(AccessFlavor access_flavor, Dtype access_type,
                         std::string var_name, std::string src_ptr,
                         std::string offset, bool define_variable,
                         std::string pred, std::string default_value) {
  if (access_flavor == AccessFlavor::REGULAR) {
    return Memory::read_regular(access_type, var_name, src_ptr, offset,
                                define_variable, pred, default_value);
  }

  if (access_flavor == AccessFlavor::EXPLICIT_PTX) {
    return Memory::read_explicit_ptx(access_type, var_name, src_ptr, offset,
                                     define_variable, pred, default_value);
  }

  if (access_flavor == AccessFlavor::EXPLICIT_PTX_EVICT_LAST) {
    return Memory::read_explicit_ptx_evict_last(access_type, var_name, src_ptr,
                                                offset, define_variable, pred,
                                                default_value);
  }

  if (access_flavor == AccessFlavor::STRONG_TYPED) {
    return Memory::read_regular_strong_typed_1d_array(
        access_type, var_name, src_ptr, offset, define_variable, pred,
        default_value);
  }

  LOG(FATAL) << "Unsupported access flavor "
             << Memory::AccessFlavorToString(access_flavor);
}

std::string Memory::write(AccessFlavor access_flavor, Dtype access_type,
                          std::string var_name, std::string dst_ptr,
                          std::string offset, std::string pred) {
  if (access_flavor == AccessFlavor::REGULAR) {
    return Memory::write_regular(access_type, var_name, dst_ptr, offset, pred);
  }

  if (access_flavor == AccessFlavor::EXPLICIT_PTX) {
    return Memory::write_explicit_ptx(access_type, var_name, dst_ptr, offset,
                                      pred);
  }

  if (access_flavor == AccessFlavor::STRONG_TYPED) {
    return Memory::write_regular_strong_typed_1d_array(access_type, var_name,
                                                       dst_ptr, offset, pred);
  }

  LOG(FATAL) << "Unsupported access flavor "
             << Memory::AccessFlavorToString(access_flavor);
}

std::string Memory::broadcast_read(Dtype access_type, std::string var_name,
                                   std::string src_ptr, std::string offset,
                                   bool define_variable, std::string pred,
                                   std::string default_value) {
  std::stringstream ss;

  ss << "// Broadcast read;\n";

  if (define_variable)
    ss << access_type.to_string() << " " << var_name << ";\n";

  EXPECT_TRUE(access_type.is_vectorized(),
              "Broadcast read should be applied on vectorized data type.");

  std::string offset_ptr = mononn_engine::helpers::string_format(
      "(&reinterpret_cast<%s *>(%s)[%s])",
      access_type.get_primitive_type().to_string().c_str(), src_ptr.c_str(),
      offset.c_str());

  ss << mononn_engine::helpers::string_format(
      "cutlass::arch::global_load_broadcast<%s, %d>(%s, (void *)%s, %s);\n",
      access_type.get_primitive_type().to_string().c_str(),
      access_type.get_elements_per_access(), var_name.c_str(),
      offset_ptr.c_str(), pred.c_str());

  if (pred != "true") {
    std::string typed_default_value = mononn_engine::helpers::string_format(
        "%s(%s)", access_type.get_primitive_type().to_string().c_str(),
        default_value.c_str());
    ss << mononn_engine::helpers::string_format("if (!(%s)) { %s = %s; }\n",
                                                pred.c_str(), var_name.c_str(),
                                                typed_default_value.c_str());
  }

  return ss.str();
}

std::string Memory::read_regular(Dtype access_type, std::string var_name,
                                 std::string src_ptr, std::string offset,
                                 bool define_variable, std::string pred,
                                 std::string default_value) {
  std::stringstream ss;

  if (define_variable) ss << access_type.to_string() << " ";
  ss << var_name << " = ";

  std::string value_on_load = mononn_engine::helpers::string_format(
      "reinterpret_cast<%s *>(%s)[%s]", access_type.to_string().c_str(),
      src_ptr.c_str(), offset.c_str());
  if (pred == "true") {
    ss << value_on_load << ";";
  } else {
    std::string typed_default_value = mononn_engine::helpers::string_format(
        "%s(%s)", access_type.get_primitive_type().to_string().c_str(),
        default_value.c_str());
    ss << mononn_engine::helpers::string_format(
        "(%s) ? (%s) : (%s);", pred.c_str(), value_on_load.c_str(),
        typed_default_value.c_str());
  }

  ss << "\n";

  return ss.str();
}

std::string Memory::read_explicit_ptx(Dtype access_type, std::string var_name,
                                      std::string src_ptr, std::string offset,
                                      bool define_variable, std::string pred,
                                      std::string default_value) {
  std::stringstream ss;

  if (define_variable) {
    ss << access_type.to_string() << " " << var_name << ";"
       << "\n";
  }

  std::string offset_ptr = mononn_engine::helpers::string_format(
      "(&reinterpret_cast<%s *>(%s)[%s])", access_type.to_string().c_str(),
      src_ptr.c_str(), offset.c_str());

  ss << mononn_engine::helpers::string_format(
      "cutlass::arch::global_load<%s, sizeof(%s)>(%s, (void *)%s, %s);\n",
      access_type.to_string().c_str(), access_type.to_string().c_str(),
      var_name.c_str(), offset_ptr.c_str(), pred.c_str());

  if (pred != "true") {
    std::string typed_default_value = mononn_engine::helpers::string_format(
        "%s(%s)", access_type.get_primitive_type().to_string().c_str(),
        default_value.c_str());
    ss << mononn_engine::helpers::string_format("if (!(%s)) { %s = %s; }\n",
                                                pred.c_str(), var_name.c_str(),
                                                typed_default_value.c_str());
  }

  return ss.str();
}

std::string Memory::read_explicit_ptx_evict_last(
    Dtype access_type, std::string var_name, std::string src_ptr,
    std::string offset, bool define_variable, std::string pred,
    std::string default_value) {
  std::stringstream ss;

  if (define_variable) {
    ss << access_type.to_string() << " " << var_name << ";"
       << "\n";
  }

  std::string offset_ptr = mononn_engine::helpers::string_format(
      "(&reinterpret_cast<%s *>(%s)[%s])", access_type.to_string().c_str(),
      src_ptr.c_str(), offset.c_str());

  ss << mononn_engine::helpers::string_format(
      "cutlass::arch::global_load_evict_last<%s, sizeof(%s)>(%s, (void *)%s, "
      "%s);\n",
      access_type.to_string().c_str(), access_type.to_string().c_str(),
      var_name.c_str(), offset_ptr.c_str(), pred.c_str());

  if (pred != "true") {
    std::string typed_default_value = mononn_engine::helpers::string_format(
        "%s(%s)", access_type.get_primitive_type().to_string().c_str(),
        default_value.c_str());
    ss << mononn_engine::helpers::string_format("if (!(%s)) { %s = %s; }\n",
                                                pred.c_str(), var_name.c_str(),
                                                typed_default_value.c_str());
  }

  return ss.str();
}

std::string Memory::write_regular(Dtype access_type, std::string var_name,
                                  std::string dst_ptr, std::string offset,
                                  std::string pred) {
  std::stringstream ss;

  std::string load_str = mononn_engine::helpers::string_format(
      "reinterpret_cast<%s *>(%s)[%s] = %s", access_type.to_string().c_str(),
      dst_ptr.c_str(), offset.c_str(), var_name.c_str());

  if (pred == "true") {
    ss << load_str << ";";
  } else {
    ss << mononn_engine::helpers::string_format("if (%s) { %s; }", pred.c_str(),
                                                load_str.c_str());
  }

  ss << "\n";
  return ss.str();
}

std::string Memory::write_explicit_ptx(Dtype access_type, std::string var_name,
                                       std::string dst_ptr, std::string offset,
                                       std::string pred) {
  std::stringstream ss;

  std::string ptr_offset = mononn_engine::helpers::string_format(
      "(&reinterpret_cast<%s *>(%s)[%s])", access_type.to_string().c_str(),
      dst_ptr.c_str(), offset.c_str());

  ss << mononn_engine::helpers::string_format(
      "cutlass::arch::global_store<%s, sizeof(%s)>(%s, (void *)%s, %s)",
      access_type.to_string().c_str(), access_type.to_string().c_str(),
      var_name.c_str(), ptr_offset.c_str(), pred.c_str());

  return ss.str();
}

std::string Memory::prefetch_l1(Dtype access_type, std::string ptr,
                                std::string offset) {
  std::string ptr_offset = mononn_engine::helpers::string_format(
      "(&reinterpret_cast<%s *>(%s)[%s])", access_type.to_string().c_str(),
      ptr.c_str(), offset.c_str());

  return mononn_engine::helpers::string_format(
      R"(asm volatile("prefetch.global.L1 [%0];\n" :: "l"(reinterpret_cast<void *>((void *)%s)));)"
      "\n",
      ptr_offset.c_str());
}

std::string Memory::prefetch_l1(Dtype access_type, std::string ptr,
                                std::string offset, std::string predicate) {
  std::string ptr_offset = mononn_engine::helpers::string_format(
      "(&reinterpret_cast<%s *>(%s)[%s])", access_type.to_string().c_str(),
      ptr.c_str(), offset.c_str());

  return mononn_engine::helpers::string_format(R"(
asm volatile(
        "{\n"
        " .reg .pred p;\n"
        " setp.ne.b32 p, %1, 0;\n"
        " @p prefetch.global.L1 [%0];\n"
        "}\n" :: "l"(reinterpret_cast<void *>((void *)%s)), "r"(int(%s)));)"
                                               "\n",
                                               ptr_offset.c_str(),
                                               predicate.c_str());
}

std::string Memory::prefetch_l2(Dtype access_type, std::string ptr,
                                std::string offset) {
  std::string ptr_offset = mononn_engine::helpers::string_format(
      "(&reinterpret_cast<%s *>(%s)[%s])", access_type.to_string().c_str(),
      ptr.c_str(), offset.c_str());

  return mononn_engine::helpers::string_format(
      R"(asm volatile("prefetch.global.L2::evict_last [%0];\n" :: "l"(reinterpret_cast<void *>((void *)%s)));)"
      "\n",
      ptr_offset.c_str());
}

std::string Memory::prefetch_l2(Dtype access_type, std::string ptr,
                                std::string offset, std::string predicate) {
  std::string ptr_offset = mononn_engine::helpers::string_format(
      "(&reinterpret_cast<%s *>(%s)[%s])", access_type.to_string().c_str(),
      ptr.c_str(), offset.c_str());

  return mononn_engine::helpers::string_format(R"(
asm volatile(
        "{\n"
        " .reg .pred p;\n"
        " setp.ne.b32 p, %1, 0;\n"
        " @p prefetch.global.L2::evict_last [%0];\n"
        "}\n" :: "l"(reinterpret_cast<void *>((void *)%s)), "r"(int(%s)));)"
                                               "\n",
                                               ptr_offset.c_str(),
                                               predicate.c_str());
}

std::string Memory::read_regular_strong_typed_1d_array(
    Dtype access_type, std::string var_name, std::string src_ptr,
    std::string offset, bool define_variable, std::string pred,
    std::string default_value) {
  std::stringstream ss;

  if (define_variable) ss << access_type.to_string() << " ";
  ss << var_name << " = ";

  std::string value_on_load = mononn_engine::helpers::string_format(
      "%s[%s]", src_ptr.c_str(), offset.c_str());
  if (pred == "true") {
    ss << value_on_load << ";";
  } else {
    std::string typed_default_value = mononn_engine::helpers::string_format(
        "%s(%s)", access_type.get_primitive_type().to_string().c_str(),
        default_value.c_str());
    ss << mononn_engine::helpers::string_format(
        "(%s) ? (%s) : (%s);", pred.c_str(), value_on_load.c_str(),
        typed_default_value.c_str());
  }

  ss << "\n";

  return ss.str();
}

std::string Memory::write_regular_strong_typed_1d_array(Dtype access_type,
                                                        std::string var_name,
                                                        std::string dst_ptr,
                                                        std::string offset,
                                                        std::string pred) {
  std::stringstream ss;

  std::string load_str = mononn_engine::helpers::string_format(
      "(%s)[%s] = %s", dst_ptr.c_str(), offset.c_str(), var_name.c_str());

  if (pred == "true") {
    ss << load_str << ";";
  } else {
    ss << mononn_engine::helpers::string_format("if (%s) { %s; }", pred.c_str(),
                                                load_str.c_str());
  }

  ss << "\n";
  return ss.str();
}
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine