#include "mononn_engine/core/op/cluster_op.h"
#include "tensorflow/core/platform/logging.h"
#include "mononn_engine/core/semantic/function_invocation.h"
#include "mononn_engine/config/config.h"
#include "mononn_engine/core/gpu/buffer_manager.h"
#include "mononn_engine/core/gpu/multi_buffer.h"
#include "mononn_engine/core/op_annotation/cluster_type.h"
#include "mononn_engine/core/schedule/loop.h"
#include "mononn_engine/core/op_annotation/op_attribute.h"
#include "mononn_engine/core/gpu/smem_manager.h"
#include "mononn_engine/helpers/env_variable.h"

namespace mononn_engine {
namespace core {
namespace op {
    using Schedule = mononn_engine::core::schedule::Schedule;
    using Graph = mononn_engine::core::graph::Graph;
    using OpImpl = mononn_engine::core::op_impl::OpImplBase;
    using TensorSpec = mononn_engine::core::tensor::TensorSpec;
    using Op = mononn_engine::core::op::Op;
    using FunctionInvocation = mononn_engine::core::semantic::FunctionInvocation;
    using Config = mononn_engine::config::Config;
    using BufferManager = mononn_engine::core::gpu::BufferManager;
    using MultiBuffer = mononn_engine::core::gpu::MultiBuffer;
    using ClusterType = mononn_engine::core::op_annotation::ClusterType;
    using Loop = mononn_engine::core::schedule::Loop;
    using OpAttribute = mononn_engine::core::op_annotation::OpAttribute;
    using SmemManager = mononn_engine::core::gpu::SmemManager;
    using Dtype = mononn_engine::core::tensor::Dtype;
    using EnvVar = mononn_engine::helpers::EnvVar;

    ClusterOp::ClusterOp(std::string _name, std::vector<std::shared_ptr<Op>> _operands, std::vector<TensorSpec> _output_specs) : Op(_name, _operands, _output_specs) {
        
    }

    OpType ClusterOp::get_type() const {
        return OpType::cluster;
    }
    
    std::vector<std::shared_ptr<OpImpl>> ClusterOp::generate_candidate_implementation(std::shared_ptr<CUDAContext> context, Tier tier) const {
        LOG(FATAL) << "Cluster op do not have concrete implementation";
    }

    const std::vector<std::string>& ClusterOp::get_hlo_instruction_name_list() const {
        return this->hlo_instruction_name_list;
    }

    void ClusterOp::add_hlo_instruction_name(const std::string &_hlo_instruction_name) {
        this->hlo_instruction_name_list.push_back(_hlo_instruction_name);
    }

    void ClusterOp::set_schedule(Schedule _schedule) {
        this->schedule = std::make_shared<Schedule>(_schedule);
    }

    Schedule ClusterOp::get_schedule() const {
        EXPECT_TRUE(this->schedule, "Schedule not available");
        return *this->schedule;
    }

    void ClusterOp::set_graph(std::shared_ptr<Graph> _graph) {
        this->graph = _graph;
    }

    std::shared_ptr<Graph> ClusterOp::get_graph() {
        EXPECT_TRUE(this->graph, "Graph not available");
        return this->graph;
    }

    std::shared_ptr<const Graph> ClusterOp::get_graph() const {
        EXPECT_TRUE(this->graph, "Graph not available");
        return std::static_pointer_cast<const Graph>(this->graph);
    }

    Graph *ClusterOp::get_graph_ptr() {
        EXPECT_TRUE(this->graph, "Graph not available");
        return this->graph.get();
    }

    const Graph *ClusterOp::get_graph_ptr() const {
        EXPECT_TRUE(this->graph, "Graph not available");
        return this->graph.get();
    }

    void ClusterOp::set_cluster_type(ClusterType _cluster_type) {
        LOG(FATAL) << "Cannot set cluster type for cluster node";
    }

    std::string ClusterOp::generate_cluster_invocation() const {
        FunctionInvocation invocation(this->get_name() + "_computation");
        for (auto const &operand : this->get_operands()) {
            std::string operand_name = operand->get_name();
            invocation.add_arg(mononn_engine::helpers::string_format("static_cast<void *>(%s)", BufferManager::get_buffer_name(operand_name).c_str()));
        }

        std::string output_buffer = BufferManager::get_buffer_name(this->get_name());

        auto TF_MONONN_ENABLED = EnvVar::is_true("TF_MONONN_ENABLED");

        if (this->get_graph()->get_output_nodes().size() > 1) {
            if (TF_MONONN_ENABLED) {
                for (int idx = 0; idx < this->get_output_specs_count(); ++idx) {
                    std::string arg_name = mononn_engine::helpers::string_format("get_tuple_element_%s_%d", this->get_name().c_str(), idx);
                    invocation.add_arg(arg_name);
                }
            } else {
                MultiBuffer multi_buffer(output_buffer);

                for (auto const &node_name : this->get_graph()->get_output_nodes()) {
                    std::shared_ptr<const Op> node = this->get_graph()->get_node(node_name);
                    multi_buffer.add_buffer(node->get_output_spec(0));
                }

                for (int idx = 0; idx < this->get_graph()->get_output_nodes().size(); ++idx) {
                    invocation.add_arg(multi_buffer.get_pointer_to_buffer(idx));
                }
            }
        } else {
            std::string output_node_name = this->get_graph()->get_output_nodes()[0];
            auto const output_node = this->get_graph_ptr()->get_node(output_node_name);

            if (output_node->get_output_specs_count() > 1) {
                if (output_node->get_type() != OpType::reduce) {
                    LOG(FATAL) << "Reduce node expected, got node " << output_node_name << " with type " << output_node->get_type().to_string();
                }

                for (int tuple_idx = 0; tuple_idx < output_node->get_output_specs_count(); ++tuple_idx) {
                    invocation.add_arg(mononn_engine::helpers::string_format("get_tuple_element_%s_%d", this->get_name().c_str(), tuple_idx));
                }
            } else {
                invocation.add_arg(output_buffer);
            }
        }

        std::string result = invocation.to_string() + ";";

        if (Config::get()->print_hlo_text) {
            result = "// " + this->get_hlo_text() + "\n" + result;
        }

        return result;
    }

    std::shared_ptr<OpImpl> ClusterOp::get_implementation() const {
        LOG(FATAL) << "Cluster node do not have implementation";
    }

    void ClusterOp::set_implementation(std::shared_ptr<OpImpl> _impl) {
        LOG(FATAL) << "Cluster node do not have implementation";
    }

    void ClusterOp::set_instruction_parallel_factor(int _ilp_factor) {
        this->ilp_factor = _ilp_factor;
        this->graph->set_instruction_parallel_factor(_ilp_factor);

        Schedule new_schedule;

        int n_loop_schedule = this->get_schedule().num_loop_schedule();
        for (int idx = 0; idx < n_loop_schedule; ++idx) {
            if (idx == n_loop_schedule - 1) {
                Loop loop = this->get_schedule().get_loop_schedule(idx).instruction_level_parallelism(_ilp_factor);
                new_schedule.add_loop_schedule(loop);
            } else {
                new_schedule.add_loop_schedule(this->get_schedule().get_loop_schedule(idx));
            }
        }

        new_schedule.set_locality_tier(this->schedule->get_locality_tier());
        *this->schedule = new_schedule;
    }

    void ClusterOp::trace_symbolic_index() {
        LOG(FATAL) << "Unimplemented";
    }

    // void ClusterOp::propagate_ilp_index_to_implementation() {
    //     LOG(FATAL) << "Cannot propagate index for cluster op";
    // }

    void ClusterOp::append_sub_cluster_tag(const std::string &sub_cluster_tag) {
        this->sub_cluster_tag_order.push_back(sub_cluster_tag);
    }

    void ClusterOp::append_sub_cluster_tags(const std::vector<std::string> &sub_cluster_tags) {
        this->sub_cluster_tag_order.insert(this->sub_cluster_tag_order.end(),
                                           sub_cluster_tags.begin(),
                                           sub_cluster_tags.end());
    }

    void ClusterOp::append_sub_cluster_type(const std::string &sub_cluster_type) {
        this->sub_cluster_type_order.push_back(sub_cluster_type);
    }

    void ClusterOp::append_sub_cluster_types(const std::vector<std::string> &sub_cluster_types) {
        this->sub_cluster_type_order.insert(this->sub_cluster_type_order.end(),
                                            sub_cluster_types.begin(),
                                            sub_cluster_types.end());
    }

    void ClusterOp::set_sub_cluster_tag_order(const std::vector<std::string> &_sub_cluster_tag_order) {
        this->sub_cluster_tag_order = _sub_cluster_tag_order;
    }

    void ClusterOp::set_sub_cluster_type_order(const std::vector<std::string> &_sub_cluster_type_order) {
        this->sub_cluster_type_order = _sub_cluster_type_order;
    }

    const std::vector<std::string> &ClusterOp::get_sub_cluster_tag_order() const {
        return this->sub_cluster_tag_order;
    }

    const std::vector<std::string> &ClusterOp::get_sub_cluster_type_order() const {
        return this->sub_cluster_type_order;
    }

    bool ClusterOp::is_cluster_contain_async_prefetched_nodes() const {
        for (auto const &node_name : this->graph->get_node_list_by_type(OpType::parameter)) {
            auto node = this->graph->get_node(node_name);

            if (node->has_attribute(OpAttribute::is_parameter_async_prefetched)) {
                return true;
            }
        }

        return false;
    }

    std::string ClusterOp::generate_async_pipeline_initialization() const {
        LOG(FATAL) << "Not implemented in " << this->get_cluster_type().to_string();
    }

    std::string ClusterOp::generate_async_pipeline_prefetch() const {
        LOG(FATAL) << "Not implemented in " << this->get_cluster_type().to_string();
    }

    std::string ClusterOp::generate_async_pipeline_stage_increment() const {
        const std::string stage_count_var_name = "total_stage_count";
        return mononn_engine::helpers::string_format("stage_id = (stage_id + 1) %% %s;\n", stage_count_var_name.c_str());
    }

    std::string ClusterOp::generate_async_pipeline_flush() const {
        return "asynchronous::wait_all()();\n";
    }

    void ClusterOp::set_cuda_context(std::shared_ptr<CUDAContext> _cuda_context) {
        this->cuda_context = _cuda_context;
    }

    void ClusterOp::initialize_smem_manager() {
        LOG(FATAL) << "Unimplemented";
    }

    SmemManager *ClusterOp::get_smem_manager() {
        return this->smem_manager.get();
    }

    int ClusterOp::get_horizontal_fusion_count() const {
        return this->horizontal_fusion_count;
    }

    void ClusterOp::set_horizontal_fusion_count(const int &_horizontal_fusion_count) {
        this->horizontal_fusion_count = _horizontal_fusion_count;
    }
}
}
}