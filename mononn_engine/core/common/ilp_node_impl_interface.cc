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


#include "mononn_engine/core/common/ilp_node_impl_interface.h"

namespace mononn_engine {
namespace core {
namespace common {
using ConcreteIndexStamp = mononn_engine::core::context::ConcreteIndexStamp;
using SymbolicIndexStamp = mononn_engine::core::context::SymbolicIndexStamp;

std::vector<ConcreteIndexStamp> ILPNodeImplInterface::get_ilp_concrete_index(
    int ilp_id) {
  return this->ilp_concrete_index_list[ilp_id];
}

ConcreteIndexStamp ILPNodeImplInterface::get_ilp_concrete_index(int ilp_id,
                                                                int index_id) {
  return this->ilp_concrete_index_list[ilp_id][index_id];
}
}  // namespace common
}  // namespace core
}  // namespace mononn_engine