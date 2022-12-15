#pragma once
#include <vector>
#include <memory>

#include "mononn_engine/core/op/cluster_op.h"

namespace mononn_engine {
namespace core {
namespace op {
    class ClusterReduce : public ClusterOp {
    public:
        ClusterReduce(std::string _name, std::vector<std::shared_ptr<Op>> _operands, std::vector<TensorSpec> _output_specs);
        
        TensorShape get_loop_shape() const override;
        Schedule construct_schedule(LocalityTier::Tier tier) override;

        void setup_codegen() override;
        std::string generate_cluster_code() const override;

        std::vector<Op*> get_reduce_nodes();
        std::vector<const Op*> get_reduce_nodes() const;

        std::vector<Op*> get_reduce_nodes_in_last_sub_cluster();
        std::vector<const Op*> get_reduce_nodes_in_last_sub_cluster() const;

        void trace_symbolic_index() override;

        bool is_cluster_reduce() const override;
        ClusterType get_cluster_type() const override;

        int get_reduction_dimension_size() const;

        std::string generate_async_pipeline_initialization() const override;
        std::string generate_async_pipeline_prefetch() const override;

        void initialize_smem_manager() override;
    private:
    };
}
}
}