#pragma once
#include <memory>

#include "mononn_engine/core/schedule/schedule.h"
#include "mononn_engine/core/op/cluster_op.h"

namespace mononn_engine {
namespace core {
namespace schedule {
    class Vectorizer {
    public:
        using ClusterOp = mononn_engine::core::op::ClusterOp;

        static void vectorize(std::shared_ptr<ClusterOp> cluster_op);
    private:
    };

}
}
}