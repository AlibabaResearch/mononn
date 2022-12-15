#include "mononn_engine/optimization/topology_simplification_pass.h"
#include "mononn_engine/core/op/cluster_op.h"
#include "mononn_engine/core/op/concatenate.h"
#include "mononn_engine/core/op/slice.h"
#include "mononn_engine/optimization/common.h"
#include "mononn_engine/helpers/string_helpers.h"

namespace mononn_engine {
namespace optimization {
    using OpType = mononn_engine::core::op::OpType;
    using Concatenate = mononn_engine::core::op::Concatenate;
    using Slice = mononn_engine::core::op::Slice;
    using Op = mononn_engine::core::op::Op;
    using ClusterOp = mononn_engine::core::op::ClusterOp;
    using Graph = mononn_engine::core::graph::Graph;
    using Edge = mononn_engine::core::edge::Edge<Op>;

    std::string TopologySimplificationPass::name() const {
        return PassName::TopologySimplificationPass;
    }

    // A interim workaround for concatenate-slice pattern in CLIP model
    // horizontally_fused_computation {
    //     param_0_0 = f32[16]{0} parameter(0)
    //     rsqrt.130 = f32[16]{0} rsqrt(param_0_0), metadata={op_type="Rsqrt" op_name="StatefulPartitionedCall/clip/norm_1/Sqrt"}
    //     broadcast.2866 = f32[16,512]{1,0} broadcast(rsqrt.130), dimensions={0}, metadata={op_type="Mul" op_name="StatefulPartitionedCall/clip/truediv_1"}
    //     param_0_1 = f16[16,512]{1,0} parameter(1)
    //     convert.795 = f32[16,512]{1,0} convert(param_0_1), metadata={op_type="Cast" op_name="StatefulPartitionedCall/clip/text_projection/MatMul-0-StatefulPartitionedCall/clip/truediv_1-0-CastToFp32-AutoMixedPrecision"}
    //     multiply.1066 = f32[16,512]{1,0} multiply(broadcast.2866, convert.795), metadata={op_type="Mul" op_name="StatefulPartitionedCall/clip/truediv_1"}
    //     convert.796 = f16[16,512]{1,0} convert(multiply.1066), metadata={op_type="Cast" op_name="StatefulPartitionedCall/clip/truediv_1-0-StatefulPartitionedCall/clip/MatMul-0-CastToFp16-AutoMixedPrecision"}
    //     reshape.520 = f16[8192]{0} reshape(convert.796), metadata={op_type="Cast" op_name="StatefulPartitionedCall/clip/truediv_1-0-StatefulPartitionedCall/clip/MatMul-0-CastToFp16-AutoMixedPrecision"}
    //     param_1_0 = f32[16]{0} parameter(2)
    //     rsqrt.131 = f32[16]{0} rsqrt(param_1_0), metadata={op_type="Rsqrt" op_name="StatefulPartitionedCall/clip/norm/Sqrt"}
    //     broadcast.2867 = f32[16,512]{1,0} broadcast(rsqrt.131), dimensions={0}, metadata={op_type="Mul" op_name="StatefulPartitionedCall/clip/truediv"}
    //     param_1_1 = f16[16,512]{1,0} parameter(3)
    //     convert.798 = f32[16,512]{1,0} convert(param_1_1), metadata={op_type="Cast" op_name="StatefulPartitionedCall/clip/visual_projection/MatMul-0-StatefulPartitionedCall/clip/norm/mul-0-CastToFp32-AutoMixedPrecision"}
    //     multiply.1067 = f32[16,512]{1,0} multiply(broadcast.2867, convert.798), metadata={op_type="Mul" op_name="StatefulPartitionedCall/clip/truediv"}
    //     convert.799 = f16[16,512]{1,0} convert(multiply.1067), metadata={op_type="Cast" op_name="StatefulPartitionedCall/clip/truediv-0-StatefulPartitionedCall/clip/MatMul-1-CastToFp16-AutoMixedPrecision"}
    //     reshape.521 = f16[8192]{0} reshape(convert.799), metadata={op_type="Cast" op_name="StatefulPartitionedCall/clip/truediv-0-StatefulPartitionedCall/clip/MatMul-1-CastToFp16-AutoMixedPrecision"}
    //     concatenate.81 = f16[16384]{0} concatenate(reshape.520, reshape.521), dimensions={0}
    //     slice.365 = f16[8192]{0} slice(concatenate.81), slice={[0:8192]}
    //     slice.366 = f16[8192]{0} slice(concatenate.81), slice={[8192:16384]}
    //     ROOT tuple.75 = (f16[8192]{0}, f16[8192]{0}) tuple(slice.365, slice.366)
    // }
    void simplify_concatenate_slice_pattern(Op *candidiate_node) {
        if (candidiate_node->get_type() != OpType::cluster) {
            return;
        }

        if (!candidiate_node->as<ClusterOp>()->is_cluster_elewise()) {
            return;
        }
        
        auto try_simplify = [](Graph *graph, Concatenate *concat_node, std::vector<Slice *> slice_node_list) -> bool {

            std::sort(slice_node_list.begin(), slice_node_list.end(), 
                [concat_rank = concat_node->get_dimension()](const Slice *slice_node_a, const Slice *slice_node_b) -> bool {
                return slice_node_a->get_slice_start(concat_rank) < slice_node_b->get_slice_start(concat_rank);
            });

            for (int rank = 0; rank < concat_node->get_output_spec(0).rank(); ++rank) {
                if (rank == concat_node->get_dimension()) {
                    int cumulative = 0;
                    int slice_node_id = 0;
                    for (auto &slice_node : slice_node_list) {
                        if (slice_node->get_slice_start(rank) != cumulative) {
                            return false;
                        }

                        int slice_width = slice_node->get_slice_limit(rank) - slice_node->get_slice_start(rank);

                        if (slice_width != concat_node->get_operand(slice_node_id)->get_output_spec(0).get_shape(rank)) {
                            return false;
                        }

                        cumulative += slice_node->get_output_spec(0).get_shape(rank);
                        slice_node_id++;
                    }
                } else {
                    for (auto &slice_node : slice_node_list) {
                        int slice_width = slice_node->get_slice_limit(rank) - slice_node->get_slice_start(rank);
                        if (slice_width != concat_node->get_output_spec(0).get_shape(rank)) {

                            return false;
                        }
                    }
                }
            }
            
            if (std::all_of(slice_node_list.begin(), slice_node_list.end(), [graph](const Slice *slice_node) -> bool {
                return graph->is_output_node(slice_node->get_name());
            })) {
                
                for (auto &edge : graph->get_node_output_edges(concat_node->get_name())) {
                    graph->remove_edge(edge);
                }

                for (auto slice_node : slice_node_list) {
                    graph->remove_node(slice_node->get_name());
                }

                std::vector<std::string> concat_operands;

                for (auto edge : graph->get_node_input_edges(concat_node->get_name())) {
                    concat_operands.push_back(edge->get_src_name());
                    graph->remove_edge(edge);
                }

                graph->remove_node(concat_node->get_name());

                for (auto const &node_name : concat_operands) {
                    graph->mark_as_output_node(node_name);
                }

                return true;
            } else if (std::none_of(slice_node_list.begin(), slice_node_list.end(), [graph](const Slice *slice_node) -> bool {
                return graph->is_output_node(slice_node->get_name());
            })) {
                std::vector<std::string> name_list;

                for (auto s : slice_node_list) {
                    name_list.push_back(s->get_name());
                }


                return false; // TOOD, not support yet.
            } else {

                return false; // TODO, not support yet.
            }
        };

        ClusterOp *cluster_node = candidiate_node->as<ClusterOp>();
        Graph *graph = cluster_node->get_graph_ptr();

        for (auto const &node_name : graph->get_node_list()) {
            auto graph_node = graph->get_node(node_name);

            if (graph_node->get_type() == OpType::concatenate) {
                auto output_edges = graph->get_node_output_edges(node_name);

                if (output_edges.empty() || !std::all_of(output_edges.begin(), output_edges.end(), [](const std::shared_ptr<Edge> &edge) -> bool {
                    return edge->get_dst()->get_type() == OpType::slice;
                })) { continue; }

                Concatenate *concatenate_node = graph_node->as<Concatenate>();

                std::vector<Slice *> slice_node_list;
                std::vector<std::string> slice_node_name_list;

                for (auto &edge : output_edges) {
                    slice_node_list.push_back(edge->get_dst()->as<Slice>());
                    slice_node_name_list.push_back(edge->get_dst_name());
                }

                if (try_simplify(graph, concatenate_node, slice_node_list)) {
                    LOG(INFO) << PassName::TopologySimplificationPass << " found valid pattern simplify_concatenate_slice_pattern."
                        " Concat node " << node_name << " slice node list: " << mononn_engine::helpers::join(", ", slice_node_name_list);
                    
                    return;
                }
            }
        }
    }

    bool TopologySimplificationPass::run(Graph *graph, std::shared_ptr<CUDAContext> cuda_context) {
        for (auto const node_name : graph->get_node_list()) {
            auto node = graph->get_node(node_name);

            simplify_concatenate_slice_pattern(node.get());
        }

        return true;
    }
}
}