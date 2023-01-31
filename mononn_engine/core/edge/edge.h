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

#pragma once
#include <memory>
#include <string>

#include "mononn_engine/core/gpu/synchronization.h"

namespace mononn_engine {
namespace core {
namespace edge {
using Synchronization = mononn_engine::core::gpu::Synchronization;

template <typename OpType>
class Edge {
 public:
  Edge(std::shared_ptr<OpType> _src, std::shared_ptr<OpType> _dst)
      : src(_src), dst(_dst), sync(Synchronization::None) {}

  std::shared_ptr<OpType> get_src() const;
  std::shared_ptr<OpType> get_dst() const;
  std::string get_src_name() const;
  std::string get_dst_name() const;

  void set_sync(Synchronization _sync);
  bool need_sync() const;
  Synchronization get_sync() const;

  std::string to_string() const;

 private:
  std::shared_ptr<OpType> src, dst;
  Synchronization sync;
};
}  // namespace edge
}  // namespace core
}  // namespace mononn_engine