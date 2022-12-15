#pragma once 

#include "mononn_engine/core/op/op.h"
#include "mononn_engine/core/context/cuda_context.h"

namespace mononn_engine {
namespace codegen {
    class NodeCodegen {
    public:
        using Op = mononn_engine::core::op::Op;
        using CUDAContext = mononn_engine::core::context::CUDAContext;

        static std::string generate(std::shared_ptr<const CUDAContext> cuda_context, std::shared_ptr<const Op> node);
    private:

    };
}
}