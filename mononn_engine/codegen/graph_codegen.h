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

#include <string>

#include "mononn_engine/codegen/cuda_program.h"
#include "mononn_engine/core/context/cuda_context.h"
#include "mononn_engine/core/graph/graph.h"
#include "mononn_engine/core/op/cluster_op.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"

namespace mononn_engine {
namespace codegen {
using CUDAProgram = mononn_engine::codegen::CUDAProgram;
using CUDAContext = mononn_engine::core::context::CUDAContext;
using Graph = mononn_engine::core::graph::Graph;

class GraphCodegen {
 public:
  enum BufferManagementPolicy {
    MONONN_BUFFER_MANAGEMENT_DEFAULT,  // MonoNN self management device buffers.
    MONONN_BUFFER_MANAGEMENT_TF_XLA,  // MonoNN leverage TF XLA buffer mangement
                                      // logic.
  };

  struct Params {
    std::shared_ptr<CUDAContext> cuda_context;
    Graph* graph;
    std::unordered_set<std::string> codegen_reject_list;
    std::string kernel_name;
    std::vector<std::string> argument_list;
    bool add_model_data = true;
    bool generate_host_code = true;
    BufferManagementPolicy buffer_management_policy =
        MONONN_BUFFER_MANAGEMENT_DEFAULT;
    const std::vector<xla::BufferAllocation>*
        allocation_list;  // Optional, used for MONONN_BUFFER_MANAGEMENT_TF_XLA
    std::unordered_map<uint64_t, std::string>
        allocation_ptr_to_buffer_name;  // Optional, used for
                                        // MONONN_BUFFER_MANAGEMENT_TF_XLA
    const xla::HloModule*
        hlo_module;  // Optional, used for MONONN_BUFFER_MANAGEMENT_TF_XLA
    const xla::HloAliasAnalysis*
        alias_analysis;  // Optional, used for MONONN_BUFFER_MANAGEMENT_TF_XLA
  };

  static std::unique_ptr<CUDAProgram> generate(const Params& options);

 private:
};
}  // namespace codegen
}  // namespace mononn_engine