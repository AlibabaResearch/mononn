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

#include "mononn_engine/core/common/ilp_node_interface.h"

#include "mononn_engine/helpers/helpers.h"

namespace mononn_engine {
namespace core {
namespace common {

int ILPNodeInterface::get_instruction_parallel_factor() const {
  return this->ilp_factor;
}

//    void ILPNodeInterface::set_ilp_traced_index(int ilp_id,
//    std::vector<IndexTraceStamp> _traced_index_list) {
//        if (ilp_id > this->ilp_traced_index_list.size()) {
//            LOG(FATAL) << "ILP id " << ilp_id << " out of range. Limit " <<
//            this->ilp_traced_index_list.size();
//        }
//
//        this->ilp_traced_index_list[ilp_id] = _traced_index_list;
//    }

//    std::vector<IndexTraceStamp> ILPNodeInterface::get_ilp_traced_index(int
//    ilp_id) const {
//        if (ilp_id > this->ilp_traced_index_list.size()) {
//            LOG(FATAL) << "ILP id " << ilp_id << " out of range. Limit " <<
//            this->ilp_traced_index_list.size();
//        }
//
//        return this->ilp_traced_index_list[ilp_id];
//    }

bool ILPNodeInterface::is_instruction_parallelized() const {
  return this->ilp_factor != 1;
}

//    bool ILPNodeInterface::is_node_ilp_traced(int ilp_id) const {
//        if (this->ilp_traced_index_list.size() == 0) return false;
//
//        if (this->ilp_traced_index_list[ilp_id].size() == 0) return false;
//
//        return true;
//    }
}  // namespace common
}  // namespace core
}  // namespace mononn_engine