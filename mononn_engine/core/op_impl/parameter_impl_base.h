#pragma once
#include <memory>
#include <vector>
#include <string>

#include "mononn_engine/core/op_impl/op_impl_base.h"
#include "mononn_engine/core/tensor/tensor.h"
#include "mononn_engine/core/context/cuda_context.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
    class ParameterImplBase : public OpImplBase {
    public:
        using CUDAContext = mononn_engine::core::context::CUDAContext;
        using Tensor = mononn_engine::core::tensor::Tensor;
    private:
    };
}
}
}