#include "mononn_engine/core/op/cluster_conv_epilogue.h"
#include "tensorflow/core/platform/logging.h"
#include "mononn_engine/core/op_annotation/locality_tier.h"
#include "mononn_engine/core/op_annotation/cluster_type.h"

namespace mononn_engine {
namespace core {
namespace op {
    using Schedule = mononn_engine::core::schedule::Schedule;
    using LocalityTier = mononn_engine::core::op_annotation::LocalityTier;
    using TensorShape = mononn_engine::core::tensor::TensorShape;
    using ClusterType = mononn_engine::core::op_annotation::ClusterType;

    TensorShape ClusterConvEpilogue::get_loop_shape() const {
        LOG(FATAL) << "Loop shape undefined";
    }

    Schedule ClusterConvEpilogue::construct_schedule(LocalityTier::Tier tier){
        Schedule schedule;
        schedule.set_locality_tier(LocalityTier::kT3);

        return {schedule};
    }

    void ClusterConvEpilogue::setup_codegen() {

    }

    std::string ClusterConvEpilogue::generate_cluster_code() const {
        LOG(FATAL) << "Not implemented";
    }

    ClusterType ClusterConvEpilogue::get_cluster_type() const {
        return ClusterType::ConvEpilogue;
    }
}
}
}