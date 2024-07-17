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

#include "mononn_engine/core/op/op_type.h"

namespace mononn_engine {
namespace core {
namespace op {
std::string OpType::get_name() const { return this->name; }

std::string OpType::to_string() const { return this->name; }

#define DEFINE_OP_TYPE(op_name, op_code, ...) \
  const OpType OpType::op_code = OpType(op_name);
OP_TYPE_LIST(DEFINE_OP_TYPE)
OP_TYPE_LIST_CLUSTER(DEFINE_OP_TYPE)
OP_TYPE_LIST_ONE_FUSER_ADDON(DEFINE_OP_TYPE)

#undef DEFINE_OP_TYPE

bool OpType::operator==(const OpType& rhs) const {
  return this->name == rhs.name;
}

bool OpType::operator!=(const OpType& rhs) const {
  return this->name != rhs.name;
}

bool OpType::operator<(const OpType& rhs) const {
  return this->name < rhs.name;
}
}  // namespace op
}  // namespace core
}  // namespace mononn_engine