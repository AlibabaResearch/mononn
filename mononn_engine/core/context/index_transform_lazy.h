#pragma once
#include <vector>
#include <string>
#include <functional>
#include "mononn_engine/core/tensor/tensor_shape.h"

namespace mononn_engine {
namespace core {
namespace context {
    class IndexTransformLazy {
    public:
        using TensorShape = mononn_engine::core::tensor::TensorShape;
        static std::function<std::vector<std::string>(std::string)> offset_to_multi_index_lazy(TensorShape tensor_shape);
        static std::function<std::string(std::vector<std::string>)> multi_index_to_offset_lazy(TensorShape tensor_shape);
    private:
    };
}
}
}



