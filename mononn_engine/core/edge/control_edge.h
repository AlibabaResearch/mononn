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

#include "mononn_engine/core/op/op.h"

namespace mononn_engine {
namespace core {
namespace edge {
class ControlEdge {
 public:
  using Op = mononn_engine::core::op::Op;

  ControlEdge(std::shared_ptr<Op> _src, std::shared_ptr<Op> _dst)
      : src(_src), dst(_dst) {}

  std::shared_ptr<Op> get_src();
  std::shared_ptr<const Op> get_src() const;
  std::shared_ptr<Op> get_dst();
  std::shared_ptr<const Op> get_dst() const;

  std::string get_src_name() const;
  std::string get_dst_name() const;

  std::string to_string() const;

 private:
  std::shared_ptr<Op> src;
  std::shared_ptr<Op> dst;
};
}  // namespace edge
}  // namespace core
}  // namespace mononn_engine