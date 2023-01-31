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

#include "mononn_engine/core/edge/control_edge.h"

namespace mononn_engine {
namespace core {
namespace edge {
using Op = mononn_engine::core::op::Op;

std::shared_ptr<Op> ControlEdge::get_src() { return this->src; }

std::shared_ptr<const Op> ControlEdge::get_src() const {
  return std::static_pointer_cast<const Op>(this->src);
}

std::shared_ptr<Op> ControlEdge::get_dst() { return this->dst; }

std::shared_ptr<const Op> ControlEdge::get_dst() const {
  return std::static_pointer_cast<const Op>(this->dst);
}

std::string ControlEdge::get_src_name() const { return this->src->get_name(); }

std::string ControlEdge::get_dst_name() const { return this->dst->get_name(); }

std::string ControlEdge::to_string() const {
  return "[" + this->get_src_name() + "->" + this->get_dst_name() + "]";
}
}  // namespace edge
}  // namespace core
}  // namespace mononn_engine
