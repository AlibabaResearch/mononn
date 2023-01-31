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

#include "mononn_engine/core/edge/edge.h"
#include "mononn_engine/core/op/cluster_op.h"

namespace mononn_engine {
namespace core {
namespace edge {
using ClusterOp = mononn_engine::core::op::ClusterOp;

template <typename OpType>
std::shared_ptr<OpType> Edge<OpType>::get_src() const {
  return this->src;
}

template <typename OpType>
std::shared_ptr<OpType> Edge<OpType>::get_dst() const {
  return this->dst;
}

template <typename OpType>
std::string Edge<OpType>::get_src_name() const {
  return this->src->get_name();
}

template <typename OpType>
std::string Edge<OpType>::get_dst_name() const {
  return this->dst->get_name();
}

template <typename OpType>
void Edge<OpType>::set_sync(Synchronization _sync) {
  this->sync = _sync;
}

template <typename OpType>
bool Edge<OpType>::need_sync() const {
  return !(this->sync == Synchronization::None);
}

template <typename OpType>
Synchronization Edge<OpType>::get_sync() const {
  return this->sync;
}

template class Edge<ClusterOp>;
}  // namespace edge
}  // namespace core
}  // namespace mononn_engine