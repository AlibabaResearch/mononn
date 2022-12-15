#pragma once

#include "tensorflow/stream_executor/stream.h"
#include "mononn_engine/frontend/mononn_module.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"

namespace mononn_engine {
namespace frontend {
    using BufferAllocations = xla::gpu::BufferAllocations;
    using BufferAllocation = xla::BufferAllocation;
    using BufferAssignment = xla::BufferAssignment;

    class MonoNNModuleTuner {
    public:
        struct Params {
            stream_executor::Stream *stream; // gpu stream
            // std::vector<void *> execution_parameters; // memory buffer
            std::vector<stream_executor::DeviceMemoryBase> execution_parameters;
            const std::string kernel_name;
            const xla::HloModule *hlo_module;
            const std::vector<BufferAllocation> *allocation_list;
            const BufferAssignment *buffer_assignment;
        };

        static std::unique_ptr<MonoNNModule> tune(const Params &params);
    private:

    };
}
}