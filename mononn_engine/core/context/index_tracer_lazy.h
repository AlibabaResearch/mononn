#pragma once
#include <string>
#include <memory>
#include "mononn_engine/core/op/broadcast.h"
#include "mononn_engine/core/op/gather.h"
#include "mononn_engine/core/op/pad.h"
#include "mononn_engine/core/op/transpose.h"
#include "mononn_engine/core/op/reshape.h"
#include "mononn_engine/core/op/copy.h"
#include "mononn_engine/core/op/reduce.h"
#include "mononn_engine/core/op/slice.h"
#include "mononn_engine/core/tensor/tensor_shape.h"

namespace mononn_engine {
namespace core {
namespace context {
    class IndexTracerLazy {
    public:
        using Op = mononn_engine::core::op::Op;
        using Broadcast = mononn_engine::core::op::Broadcast;
        using Gather = mononn_engine::core::op::Gather;
        using Pad = mononn_engine::core::op::Pad;
        using Transpose = mononn_engine::core::op::Transpose;
        using Slice = mononn_engine::core::op::Slice;
        using Reduce = mononn_engine::core::op::Reduce;
        using TensorShape = mononn_engine::core::tensor::TensorShape;

        std::string get_index(std::string input_index) const;

        void trace(std::shared_ptr<const Op> op);

        void trace_broadcast(std::shared_ptr<const Broadcast> op);
        void trace_gather_operand(std::shared_ptr<const Gather> op);
        void trace_gather_indices(std::shared_ptr<const Gather> op);
        void trace_pad(std::shared_ptr<const Pad> op);
        void trace_transpose(std::shared_ptr<const Transpose> op);
        void trace_slice(std::shared_ptr<const Slice> op);
        void trace_reduce(std::shared_ptr<const Reduce> op, std::string const &inverse_reduce_dim);

    private:
        std::function<std::string(std::string)> index;
    };
}
}
}