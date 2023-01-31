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