#include <queue>
#include "mononn_engine/optimization/common.h"
#include "mononn_engine/optimization/global_synchronization_elimination_pass.h"
#include "mononn_engine/core/op/global_sync.h"
#include "mononn_engine/core/gpu/synchronization.h"
#include "mononn_engine/helpers/helpers.h"
#include "mononn_engine/core/op/op_type.h"

namespace mononn_engine {
namespace optimization {
    using OpType = mononn_engine::core::op::OpType;
    using GlobalSync = mononn_engine::core::op::GlobalSync;
    using Synchronization = mononn_engine::core::gpu::Synchronization;

    std::string GlobalSynchronizationEliminationPass::name() const {
        return PassName::GlobalSynchronizationEliminationPass;
    }

    bool GlobalSynchronizationEliminationPass::run(Graph *graph, std::shared_ptr<CUDAContext> cuda_context) {
        std::vector<std::string> seed_node =
                mononn_engine::helpers::vector_concat(graph->get_input_nodes(), graph->get_extended_input_nodes());

        struct BFSNode {
            std::string name;
            int depth;

            bool operator < (const BFSNode & rhs) const {
                return this->depth > rhs.depth;
            }
        };

//        std::priority_queue<BFSNode, std::vector<BFSNode>> q;
        std::queue<BFSNode> q;
        std::map<std::string, int> visit;

        for (auto const &node_name : seed_node) {
            q.push({node_name, 0});
        }

        while (!q.empty()) {
            BFSNode node = q.front();
            q.pop();

//            LOG(DEBUG) << node.name << " depth: " << node.depth;

            if (visit.find(node.name) == visit.end() || visit[node.name] < node.depth) {
                visit[node.name] = node.depth;
            } else {
                continue;
            }

            for (auto const &edge : graph->get_node_output_edges(node.name)) {
                if (edge->get_sync() == Synchronization::Global) {
                    q.push({edge->get_dst_name(), node.depth + 1});
                } else {
                    q.push({edge->get_dst_name(), node.depth});
                }
            }

            for (auto const &control_edge : graph->get_node_output_control_edges(node.name)) {
                q.push({control_edge->get_dst_name(), node.depth});
            }
        }

// Node f_copy_42_MI_f_copy_31_MI_f_copy_52_MI_f_copy_55_MI_f_copy_37_MI_f_copy_36_MI_f_copy_58_MI_f_copy_60_MI_f_copy_33_MI_f_
// copy_25_MI_f_copy_41_MI_f_copy_27_MI_f_copy_53_MI_f_copy_48_MI_f_copy_46_MI_f_copy_43_MI_f_copy_29_MI_f_copy_47_MI_f_copy_56_MI
// _f_copy_57_MI_f_copy_30_MI_f_copy_32_MI_f_copy_59_MI_f_copy_45_MI_f_copy_49_MI_f_copy_51_MI_f_copy_34_MI_f_copy_26_MI_f_copy_39_MI_
// f_copy_50_MI_f_copy_24_MI_f_copy_44_MI_f_copy_23_MI_f_copy_28_MI_f_copy_54_MI_f_copy_38_MI_f_copy_35_MI_f_copy_40 do not belong to any independent group

        for (auto const &node_name : graph->get_node_list()) {
            if (visit.find(node_name) == visit.end()) {
                LOG(ERROR) << "Input node list: " << mononn_engine::helpers::join(", ", graph->get_input_nodes());
                LOG(ERROR) << "Extended input node list: " << mononn_engine::helpers::join(", ", graph->get_extended_input_nodes());
                LOG(FATAL) << "Node " << node_name << " do not belong to any independent group";
            }
        }

        int max_step = 0;

        for (auto const &[node_name, step] : visit) {
            max_step = std::max(max_step, step);
        }

        std::map<int, std::vector<std::string>> node_by_step;

        for (auto const &[node_name, node_step] : visit) {
            if (node_by_step.count(node_step) == 0) {
                node_by_step[node_step] = std::vector<std::string>();
            }

            node_by_step[node_step].push_back(node_name);
        }

        // insert sync node
        for (int step = 1; step <= max_step; ++step) {
            std::string global_sync_node_name = "global_sync_" + std::to_string(step);
            std::shared_ptr<GlobalSync> global_sync = std::make_shared<GlobalSync>(global_sync_node_name);
            global_sync->set_hlo_text(global_sync_node_name);
            graph->add_node(global_sync);

            std::vector<std::string> merged_edges;

            for (auto const &node_name : node_by_step[step]) {
                graph->add_control_edge(global_sync_node_name, node_name);
            }

            for (auto const &node_name : node_by_step[step - 1]) {
                graph->add_control_edge(node_name, global_sync_node_name);
                for (auto &edge : graph->get_node_output_edges(node_name)) {
                    if (edge->get_sync() == Synchronization::Global) {
                        if (visit[edge->get_dst_name()] < step) LOG(FATAL) << "Independent group topology error";
                        merged_edges.push_back(edge->to_string());

                        edge->set_sync(Synchronization::None);
                    }
                }
            }

            EXPECT_TRUE(merged_edges.size() != 0, "No merged node for step " + std::to_string(step));
            if (merged_edges.size() > 1) LOG(INFO) << "Merge global sync: " << mononn_engine::helpers::join(" ", merged_edges) << " to " << global_sync_node_name;
        }

//        LOG(DEBUG) << "------------------------";
//
//        for (auto const &edge : graph->get_node_output_control_edges("global_sync_1")) {
//            LOG(DEBUG) << edge->to_string();
//        }
//        LOG(DEBUG) << "------------------------";
//        for (auto const &edge : graph->get_node_input_control_edges("global_sync_1")) {
//            LOG(DEBUG) << edge->to_string();
//        }
//
//        for (int step = 0; step <= max_step; ++step) {
//            std::vector<std::string> node_list;
//            for (auto const &[node_name, max_step] : visit) {
//                OpType node_type = graph->get_node(node_name)->get_type();
//                if (step == max_step && node_type != OpType::parameter && node_type != OpType::constant && node_type != OpType::get_tuple_element) {
//                    node_list.push_back(node_name);
//                }
//            }
//
//            LOG(DEBUG) << "step" << step << " node list: " << mononn_engine::helpers::join(" ", node_list);
//        }


        return true;
    }
}
}
