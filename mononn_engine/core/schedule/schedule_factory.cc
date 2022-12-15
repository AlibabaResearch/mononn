#include <vector>

#include "mononn_engine/core/schedule/schedule_factory.h"
#include "mononn_engine/core/tensor/scalar.h"
#include "mononn_engine/core/tensor/dtype.h"
#include "mononn_engine/core/schedule/loop.h"
#include "mononn_engine/core/gpu/defined.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace core {
namespace schedule {
    using Scalar = mononn_engine::core::tensor::Scalar;
    using Dtype = mononn_engine::core::tensor::Dtype;
    using CUDADefined = mononn_engine::core::gpu::CUDADefined;

    Schedule ScheduleFactory::get_schedule(LocalityTier::Tier tier, TensorShape shape) {
        if (tier == LocalityTier::kT0)
        {
            Schedule one_level_schedule;
            Scalar idx("idx", Dtype::from_string("int32"));

            Loop one_level_loop(
                shape,
                idx,
                CUDADefined::threadIdx_x_global,
                CUDADefined::threadCnt_x_global
            );

            one_level_schedule.add_loop_schedule(one_level_loop);
            one_level_schedule.set_locality_tier(Schedule::LocalityTier::kT0);

            return one_level_schedule;
        } else if (tier == LocalityTier::kT1)
        {
            Schedule two_level_warp;

            Scalar idx_i("idx_i", Dtype::from_string("int32"));
            Scalar idx_j("idx_j", Dtype::from_string("int32"));

            TensorShape loop1_shape = shape.slice_dim(0, -2);
            TensorShape loop2_shape = shape.slice_dim(-1, -1);

            Loop two_level_warp_loop1(
                loop1_shape,
                idx_i,
                CUDADefined::warp_global_id,
                CUDADefined::warp_global_count    
            );

            Loop two_level_warp_loop2(
                loop2_shape,
                idx_j,
                CUDADefined::lane_id,
                CUDADefined::warpSize
            );

            two_level_warp.add_loop_schedule(two_level_warp_loop1);
            two_level_warp.add_loop_schedule(two_level_warp_loop2);
            two_level_warp.set_locality_tier(Schedule::LocalityTier::kT1);

            return two_level_warp;
        } else if (tier == LocalityTier::kT2) {
            Scalar idx_i("idx_i", Dtype::from_string("int32"));
            Scalar idx_j("idx_j", Dtype::from_string("int32"));

            TensorShape loop1_shape = shape.slice_dim(0, -2);
            TensorShape loop2_shape = shape.slice_dim(-1, -1);

            Schedule two_level_block;
            Loop two_level_block_loop1(
                    loop1_shape,
                    idx_i,
                    CUDADefined::blockIdx_x,
                    CUDADefined::gridDim_x
            );

            Loop two_level_block_loop2(
                    loop2_shape,
                    idx_j,
                    CUDADefined::threadIdx_x,
                    CUDADefined::blockDim_x
            );

            two_level_block.add_loop_schedule(two_level_block_loop1);
            two_level_block.add_loop_schedule(two_level_block_loop2);
            two_level_block.set_locality_tier(Schedule::LocalityTier::kT2);

            return two_level_block;
        } else if (tier == LocalityTier::kT3)
        { // gemm schedule
            Schedule schedule;
            schedule.set_locality_tier(Schedule::LocalityTier::kT3);

            return {schedule};
        }

        LOG(FATAL) << "Unsupported locality tier: " << tier.to_string();
    }
}
}
}