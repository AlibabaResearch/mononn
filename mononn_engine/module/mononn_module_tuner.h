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

#include "mononn_engine/module/mononn_module.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/stream_executor/stream.h"

namespace mononn_engine {
namespace module {
using BufferAllocations = xla::gpu::BufferAllocations;
using BufferAllocation = xla::BufferAllocation;
using BufferAssignment = xla::BufferAssignment;

class MonoNNModuleTuner {
 public:
  struct Params {
    stream_executor::Stream* stream;  // gpu stream
    // std::vector<void *> execution_parameters; // memory buffer
    std::vector<stream_executor::DeviceMemoryBase> execution_parameters;
    const std::string kernel_name;
    const xla::HloModule* hlo_module;
    const std::vector<BufferAllocation>* allocation_list;
    const BufferAssignment* buffer_assignment;
  };

  static std::unique_ptr<MonoNNModule> tune(const Params& params);

 private:
};
}  // namespace module
}  // namespace mononn_engine