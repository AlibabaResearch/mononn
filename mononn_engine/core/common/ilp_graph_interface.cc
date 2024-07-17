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

#include "mononn_engine/core/common/ilp_graph_interface.h"

#include "mononn_engine/helpers/helpers.h"

namespace mononn_engine {
namespace core {
namespace common {
using IndexTraceStamp = mononn_engine::core::context::IndexTraceStamp;

int ILPGraphInterface::get_instruction_parallel_factor() {
  return this->ilp_factor;
}

bool ILPGraphInterface::is_instruction_parallelized() const {
  return this->ilp_factor != 1;
}

void ILPGraphInterface::set_ilp_traced_index(
    int ilp_id, std::string node_name,
    std::vector<IndexTraceStamp> _traced_index_list) {
  if (this->ilp_traced_index.find(node_name) == this->ilp_traced_index.end()) {
    this->ilp_traced_index[node_name] =
        std::vector<std::vector<IndexTraceStamp>>(this->ilp_factor);
  }

  this->ilp_traced_index[node_name][ilp_id] = _traced_index_list;
}

std::vector<IndexTraceStamp> ILPGraphInterface::get_ilp_traced_index(
    int ilp_id, std::string node_name) const {
  EXPECT_TRUE(this->ilp_traced_index.count(node_name) > 0,
              "Index " + node_name + " not traced");
  EXPECT_TRUE(this->ilp_traced_index.at(node_name).size() > ilp_id,
              "Ilp id " + std::to_string(ilp_id) + " out of range. Limit" +
                  std::to_string(this->ilp_traced_index.at(node_name).size()));

  return this->ilp_traced_index.at(node_name)[ilp_id];
}

bool ILPGraphInterface::is_node_ilp_traced(std::string node_name,
                                           int ilp_id) const {
  if (this->ilp_traced_index.find(node_name) == this->ilp_traced_index.end()) {
    return false;
  }

  if (this->ilp_traced_index.at(node_name).size() == 0) {
    return false;
  }

  EXPECT_TRUE(this->ilp_traced_index.at(node_name).size() == this->ilp_factor,
              "Ilp factor " + std::to_string(this->ilp_factor) + " with " +
                  std::to_string(this->ilp_traced_index.size()) +
                  "traced index");

  return !this->ilp_traced_index.at(node_name)[ilp_id].empty();
}

void ILPGraphInterface::reset_ilp_traced_index() {
  this->ilp_traced_index.clear();
}
}  // namespace common
}  // namespace core
}  // namespace mononn_engine