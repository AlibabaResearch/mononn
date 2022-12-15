#pragma once
#include "mononn_engine/core/op/cluster_op.h"
#include "mononn_engine/core/gpu/dim3.h"

namespace mononn_engine {
namespace core {
namespace op {
    class ClusterElewise : public ClusterOp {
    public:
        using Dim3 = mononn_engine::core::gpu::Dim3;

        ClusterElewise(std::string _name, std::vector<std::shared_ptr<Op>> _operands, std::vector<TensorSpec> _output_specs);

        TensorShape get_loop_shape() const override;
        Schedule construct_schedule(LocalityTier::Tier tier) override;

        void setup_codegen() override;
        std::string generate_cluster_code() const override;

        void trace_symbolic_index() override;

        bool is_cluster_elewise() const override;
        ClusterType get_cluster_type() const override;

        std::string generate_async_pipeline_initialization() const override;
        std::string generate_async_pipeline_prefetch() const override;
    private:
    
    };
}
}
}