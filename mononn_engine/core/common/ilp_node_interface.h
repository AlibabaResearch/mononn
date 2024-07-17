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

#include <vector>

namespace mononn_engine {
namespace core {
namespace common {
class ILPNodeInterface {
 public:
  virtual void set_instruction_parallel_factor(int _ilp_factor) = 0;
  int get_instruction_parallel_factor() const;
  //        virtual void set_ilp_traced_index(int ilp_id,
  //        std::vector<IndexTraceStamp> _traced_index_list); virtual
  //        std::vector<IndexTraceStamp> get_ilp_traced_index(int ilp_id) const;
  bool is_instruction_parallelized() const;
  //        bool is_node_ilp_traced(int ilp_id) const;

 protected:
  //        std::vector<std::vector<SymbolicIndexStamp>> ilp_traced_index_list;
  int ilp_factor = 1;
};
}  // namespace common
}  // namespace core
}  // namespace mononn_engine
