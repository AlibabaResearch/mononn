#pragma once
#include <vector>
#include <string>
#include <memory>

#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/op/op.h"
#include "mononn_engine/core/graph/graph.h"
#include "mononn_engine/core/op/cluster_elewise.h"
#include "mononn_engine/core/op/cluster_reduce.h"
#include "mononn_engine/core/op_annotation/cluster_type.h"

namespace mononn_engine {
namespace core {
namespace graph {
    using Op = mononn_engine::core::op::Op;
    using Graph = mononn_engine::core::graph::Graph;
    using ClusterOp = mononn_engine::core::op::ClusterOp;
    using ClusterElewise = mononn_engine::core::op::ClusterElewise;
    using ClusterReduce = mononn_engine::core::op::ClusterReduce;
    using OpType = mononn_engine::core::op::OpType;
    using ClusterType = mononn_engine::core::op_annotation::ClusterType;

    class ClusterUtil {
    public:
        template<typename T>
        static std::shared_ptr<T> merge_independent(std::string cluster_name, std::shared_ptr<ClusterOp> cluster1, std::shared_ptr<ClusterOp> cluster2);
        template<typename T>
        static std::shared_ptr<T> merge_sequential(
                std::string cluster_name,
                std::shared_ptr<ClusterOp> begin_cluster,
                std::vector<std::shared_ptr<Op>> nodes_in_path,
                std::shared_ptr<ClusterOp> end_cluster,
                Graph *graph);

        // Replace (merge) nodes in node_list with a functional equivalent one.
        // Delete old nodes in node_list in the graph and add new_node to the graph.
        // Maintain GTE node if necessary.
        static void replace_node(Graph *graph, std::vector<std::string> node_list, std::shared_ptr<Op> new_node);
        static void summary_graph(Graph *graph);

        static std::unique_ptr<Graph> deep_copy_graph(const Graph *graph);
        
    private:
    };

    template<typename T>
    class ClusterTypeOf;

    template<>
    class ClusterTypeOf<ClusterElewise> {
    public:
        static ClusterType const Type;
    };

    template<>
    class ClusterTypeOf<ClusterReduce> {
    public:
        static ClusterType const Type;
    };
}
}
}