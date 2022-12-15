#include "mononn_engine/tuning/tuning_space_generator.h"
#include "mononn_engine/parser/ir_parser_fused.h"
#include "mononn_engine/core/gpu/dim3.h"
#include "mononn_engine/core/context/cuda_context.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/helpers/protobuf.h"
#include "mononn_engine/core/op_impl/gemm_impl.h"
#include "mononn_engine/core/op_impl/conv_impl.h"
#include "tensorflow/mononn_extra/proto/gemm_specification.pb.h"
#include "tensorflow/mononn_extra/proto/conv_specification.pb.h"
#include "tensorflow/mononn_extra/proto/cluster_elewise_specification.pb.h"
#include "tensorflow/mononn_extra/proto/cluster_reduce_specification.pb.h"
#include "tensorflow/mononn_extra/proto/cutlass_config.pb.h"
#include "tensorflow/mononn_extra/proto/gemm_backend_config.pb.h"
#include "mononn_engine/optimization/intra_op_reschedule_pass.h"
#include "mononn_engine/config/config.h"
#include "mononn_engine/optimization/common.h"

namespace mononn_engine {
namespace tuning {
    using Dim3 = mononn_engine::core::gpu::Dim3;
    using Graph = mononn_engine::core::graph::Graph;
    using GraphSpecification = tensorflow::mononn_extra::proto::GraphSpecification;
    using CUDAContext = mononn_engine::core::context::CUDAContext;
    using CUDARuntimeContext = mononn_engine::core::context::CUDARuntimeContext;
    using CUDADeviceContext = mononn_engine::core::context::CUDADeviceContext;
    using OpType = mononn_engine::core::op::OpType;
    namespace proto = tensorflow::mononn_extra::proto;
    using Op = mononn_engine::core::op::Op;
    using ClusterOp = mononn_engine::core::op::ClusterOp;
    using GemmImpl = mononn_engine::core::op_impl::GemmImpl;
    using ConvImpl = mononn_engine::core::op_impl::ConvImpl;
    using IntraOpReschedulePass = mononn_engine::optimization::IntraOpReschedulePass;
    using LocalityTier = mononn_engine::core::op_annotation::LocalityTier;
    using Config = mononn_engine::config::Config;
    using PassName = mononn_engine::optimization::PassName;

    std::vector<std::unique_ptr<GraphSpecification>> TuningSpaceGenerator::generate_tuning_space(
            Graph *graph,
            std::string hlo_proto_file,
            std::vector<std::string> feeds,
            std::vector<std::string> input_data_files,
            std::vector<std::string> fetches) {
        CUDADeviceContext cuda_device_context = CUDADeviceContext::get_cuda_device_context();

        std::vector<std::unique_ptr<GraphSpecification>> tuning_space;

    //    std::vector<int> block_per_sm_space = {1, 2, 3, 4, 5};
    //    std::vector<int> block_per_sm_space = {1, 2, 3, 4};
    //    std::vector<int> thread_per_block_space = {128, 256};
        
        std::vector<int> block_per_sm_space = {1, 2, 3};
        std::vector<int> thread_per_block_space = {128, 256};

        if (Config::get()->faster_tuning) {
            block_per_sm_space = {1, 2, 3};
            thread_per_block_space = {128};
        }

        if (Config::get()->fastest_tuning) {
            block_per_sm_space = {2};
            thread_per_block_space = {128};
        }

        for (auto const &block_per_sm : block_per_sm_space) {
            for (auto const &thread_per_block : thread_per_block_space) {
                Dim3 grid_dim(cuda_device_context.sm_count * block_per_sm, 1, 1);
                Dim3 block_dim(thread_per_block, 1, 1);

                std::shared_ptr<CUDAContext> cuda_context = std::make_shared<CUDAContext>(CUDAContext::get_cuda_context(grid_dim, block_dim, "cuda_stream", cuda_device_context));

                if (cuda_context->cuda_runtime_context.smem_size < 0) { continue; }

                std::vector<std::unique_ptr<GraphSpecification>> tuning_space_for_context = std::move(TuningSpaceGenerator::generate_tuning_space(graph, cuda_context, hlo_proto_file, feeds, input_data_files, fetches));

                for (auto &space : tuning_space_for_context) {
                    tuning_space.push_back(std::move(space));
                }
            }
        }

        return std::move(tuning_space);
    }

    std::vector<std::unique_ptr<GraphSpecification>>
    TuningSpaceGenerator::generate_tuning_space(
            Graph *graph,
            std::shared_ptr<CUDAContext> cuda_context,
            std::string hlo_proto_file,
            std::vector<std::string> feeds,
            std::vector<std::string> input_data_files,
            std::vector<std::string> fetches) {
        std::vector<std::unique_ptr<GraphSpecification>> tuning_space;
        std::unique_ptr<GraphSpecification> default_graph_spec = TuningSpaceGenerator::get_default_graph_specification(graph, cuda_context, hlo_proto_file, feeds, input_data_files, fetches);
        LOG(INFO) << "Fast tuning for warp and block reduction.";

        if (!default_graph_spec) return {};

        std::vector<int> ilp_factor_space = Config::get()->candidate_ilp_factor;
        if (Config::get()->faster_tuning) {
            ilp_factor_space = {1, 2, 4, 8, 16, 32};
        }

        if (Config::get()->fastest_tuning) {
            ilp_factor_space = {1, 2};
        }

        if (std::find(Config::get()->optimization_disabled_pass.begin(), 
            Config::get()->optimization_disabled_pass.end(),
            PassName::IntraOpReschedulePass) != Config::get()->optimization_disabled_pass.end()) {
                
            ilp_factor_space = {1};
            LOG(INFO) << "TuningSpaceGenerator: " << PassName::IntraOpReschedulePass << " disabled.";
        }

        // mutate for each cluster node;
        for (auto const &cluster_node_name : graph->get_node_list_by_type(OpType::cluster)) {
            std::shared_ptr<Op> cluster_node = graph->get_node(cluster_node_name);

            if (cluster_node->is_cluster_elewise()) {
                cluster_node->as<ClusterOp>()->set_schedule(cluster_node->as<ClusterOp>()->construct_schedule(LocalityTier::kT0));
                for (auto const &ilp_factor : ilp_factor_space) {
                    if (!IntraOpReschedulePass::can_rescheduled_with_ilp_factor(cuda_context, cluster_node->as<ClusterOp>(), ilp_factor)) continue;
                    proto::ClusterElewiseSpecification cluster_elewise_spec;
                    cluster_elewise_spec.set_name(cluster_node_name);
                    cluster_elewise_spec.set_ilp_factor(ilp_factor);

                    std::unique_ptr<GraphSpecification> graph_spec = mononn_engine::helpers::deep_copy_graph_specification(default_graph_spec.get());

                    (*graph_spec->mutable_cluster_elewise_spec())[cluster_node_name] = cluster_elewise_spec;

                    TuningSpaceGenerator::set_allow_and_reject_list(graph_spec.get(), graph, cluster_node_name);

                    tuning_space.push_back(std::move(graph_spec));
                }
            } else if (cluster_node->is_cluster_reduce()) {
                for (auto const &ilp_factor : ilp_factor_space) {
                    auto tier1_schedule = cluster_node->as<ClusterOp>()->construct_schedule(LocalityTier::kT1);
                    auto tier2_schedule = cluster_node->as<ClusterOp>()->construct_schedule(LocalityTier::kT2);

                    cluster_node->as<ClusterOp>()->set_schedule(tier1_schedule);
                    if (IntraOpReschedulePass::can_rescheduled_with_ilp_factor(cuda_context, cluster_node->as<ClusterOp>(), ilp_factor)) {
                        // std::vector<int> reduce_warp_impl_space = {0, 1};
                        std::vector<int> reduce_warp_impl_space = {1};
                        if (Config::get()->faster_tuning || Config::get()->fastest_tuning) {
                            reduce_warp_impl_space = {0};
                        }

                        for (auto const &reduce_impl_id : reduce_warp_impl_space) {
                            proto::ClusterReduceSpecification cluster_reduce_spec;
                            cluster_reduce_spec.set_name(cluster_node_name);
                            cluster_reduce_spec.set_ilp_factor(ilp_factor);
                            cluster_reduce_spec.set_locality_tier(1);
                            cluster_reduce_spec.set_reduce_implementation(reduce_impl_id);

                            std::unique_ptr<GraphSpecification> graph_spec = mononn_engine::helpers::deep_copy_graph_specification(default_graph_spec.get());
                            (*graph_spec->mutable_cluster_reduce_spec())[cluster_node_name] = cluster_reduce_spec;

                            TuningSpaceGenerator::set_allow_and_reject_list(graph_spec.get(), graph, cluster_node_name);
                            tuning_space.push_back(std::move(graph_spec));
                        }
                    }

                    cluster_node->as<ClusterOp>()->set_schedule(tier2_schedule);
                    if (IntraOpReschedulePass::can_rescheduled_with_ilp_factor(cuda_context, cluster_node->as<ClusterOp>(), ilp_factor)) {
                        std::vector<int> reduce_block_impl_space = {1};
                        // std::vector<int> reduce_block_impl_space = {0, 1, 2, 3};

                        if (Config::get()->faster_tuning) {
                            reduce_block_impl_space = {0, 1};
                        }

                        if (Config::get()->fastest_tuning) {
                            reduce_block_impl_space = {0};
                        }

                        for (auto const &reduce_impl_id : reduce_block_impl_space) {
                            proto::ClusterReduceSpecification cluster_reduce_spec;
                            cluster_reduce_spec.set_name(cluster_node_name);
                            cluster_reduce_spec.set_ilp_factor(ilp_factor);
                            cluster_reduce_spec.set_locality_tier(2);
                            cluster_reduce_spec.set_reduce_implementation(reduce_impl_id);

                            std::unique_ptr<GraphSpecification> graph_spec = mononn_engine::helpers::deep_copy_graph_specification(default_graph_spec.get());
                            (*graph_spec->mutable_cluster_reduce_spec())[cluster_node_name] = cluster_reduce_spec;

                            TuningSpaceGenerator::set_allow_and_reject_list(graph_spec.get(), graph, cluster_node_name);

                            tuning_space.push_back(std::move(graph_spec));
                        }
                    }
                }
            } else {
                LOG(FATAL) << "";
            }
        }

        // mutate for each gemm node
        for (auto const &node_name : graph->get_node_list_by_type(OpType::custom_call)) {
            std::shared_ptr<Op> node = graph->get_node(node_name);

            if (node->is_gemm()) {
                auto candidate_impl = node->generate_candidate_implementation(cuda_context, 3);

                if (candidate_impl.empty()) {
                    LOG(FATAL) << "No available implementation for GEMM: " << node_name;
                }

                for (auto const &gemm_impl : candidate_impl) {
                    tensorflow::mononn_extra::proto::GemmSpecification gemm_spec;
                    gemm_spec.set_allocated_cutlass_config(gemm_impl->as<GemmImpl>()->get_cutlass_config().ConvertToProto().release());
                    gemm_spec.set_allocated_gemm_backend_config(gemm_impl->as<GemmImpl>()->get_gemm_backend_config().ConvertToProto().release());

                    std::unique_ptr<GraphSpecification> graph_spec = mononn_engine::helpers::deep_copy_graph_specification(default_graph_spec.get());
                    (*graph_spec->mutable_gemm_spec_list())[node_name] = gemm_spec;
                    TuningSpaceGenerator::set_allow_and_reject_list(graph_spec.get(), graph, node_name);

                    tuning_space.push_back(std::move(graph_spec));
                }
            } else if (node->is_conv()) {
                auto candidate_impl = node->generate_candidate_implementation(cuda_context, 3);

                if (candidate_impl.empty()) {
                    LOG(FATAL) << "No available implementation for Conv: " << node_name;
                }

                for (auto const &conv_impl : candidate_impl) {
                    tensorflow::mononn_extra::proto::ConvSpecification conv_spec;
                    conv_spec.set_allocated_cutlass_config(conv_impl->as<ConvImpl>()->get_cutlass_config().ConvertToProto().release());
                    conv_spec.set_allocated_conv_backend_config(conv_impl->as<ConvImpl>()->get_conv_backend_config().ConvertToProto().release());

                    std::unique_ptr<GraphSpecification> graph_spec = mononn_engine::helpers::deep_copy_graph_specification(default_graph_spec.get());
                    (*graph_spec->mutable_conv_spec_list())[node_name] = conv_spec;
                    TuningSpaceGenerator::set_allow_and_reject_list(graph_spec.get(), graph, node_name);

                    tuning_space.push_back(std::move(graph_spec));
                }
            } else {
                LOG(FATAL) << "";
            }
        }

        return std::move(tuning_space);
    }

    std::unique_ptr<GraphSpecification>
    TuningSpaceGenerator::get_default_graph_specification(
            Graph *graph,
            std::shared_ptr<CUDAContext> cuda_context,
            std::string hlo_proto_file,
            std::vector<std::string> feeds,
            std::vector<std::string> input_data_files,
            std::vector<std::string> fetches) {
        std::unique_ptr<GraphSpecification> default_graph_specification = std::make_unique<GraphSpecification>();

        default_graph_specification->set_allocated_cuda_context(cuda_context->ConvertToProto().release());
        default_graph_specification->set_hlo_module_proto_file(hlo_proto_file);

        for (auto const &feed : feeds) {
            default_graph_specification->add_feeds(feed);
        }

        for (auto const &data_file : input_data_files) {
            default_graph_specification->add_input_data_files(data_file);
        }

        for (auto const &fetch : fetches) {
            default_graph_specification->add_fetches(fetch);
        }

        // cluster tuning space
        for (auto const &cluster_node_name : graph->get_node_list_by_type(OpType::cluster)) {
            std::shared_ptr<Op> cluster_node = graph->get_node(cluster_node_name);
            if (cluster_node->is_cluster_elewise()) {
                proto::ClusterElewiseSpecification cluster_elewise_spec;
                cluster_elewise_spec.set_name(cluster_node_name);
                cluster_elewise_spec.set_ilp_factor(1);
                (*default_graph_specification->mutable_cluster_elewise_spec())[cluster_node_name] = cluster_elewise_spec;
            } else if (cluster_node->is_cluster_reduce()) {
                proto::ClusterReduceSpecification cluster_reduce_spec;
                cluster_reduce_spec.set_name(cluster_node_name);
                cluster_reduce_spec.set_ilp_factor(1);
                cluster_reduce_spec.set_locality_tier(1);
                cluster_reduce_spec.set_reduce_implementation(0);
                (*default_graph_specification->mutable_cluster_reduce_spec())[cluster_node_name] = cluster_reduce_spec;
            } else {
                LOG(FATAL) << "";
            }
        }

        // gemm + conv tuning space
        for (auto const &node_name : graph->get_node_list_by_type(OpType::custom_call)) {
            std::shared_ptr<Op> node = graph->get_node(node_name);

            if (node->is_gemm()) {
                proto::GemmSpecification gemm_spec;
                auto candidate_impl = node->generate_candidate_implementation(cuda_context, 3);

                if (candidate_impl.empty()) return nullptr;

                GemmImpl *gemm_impl = candidate_impl[0]->as<GemmImpl>();
                gemm_spec.set_allocated_cutlass_config(gemm_impl->get_cutlass_config().ConvertToProto().release());
                gemm_spec.set_allocated_gemm_backend_config(gemm_impl->get_gemm_backend_config().ConvertToProto().release());

                (*default_graph_specification->mutable_gemm_spec_list())[node_name] = gemm_spec;
            } else if (node->is_conv()) {
                proto::ConvSpecification conv_spec;
                auto candidate_impl = node->generate_candidate_implementation(cuda_context, 3);

                if (candidate_impl.empty()) return nullptr;

                ConvImpl *conv_impl = candidate_impl[0]->as<ConvImpl>();
                conv_spec.set_allocated_cutlass_config(conv_impl->get_cutlass_config().ConvertToProto().release());
                conv_spec.set_allocated_conv_backend_config(conv_impl->get_conv_backend_config().ConvertToProto().release());

                (*default_graph_specification->mutable_conv_spec_list())[node_name] = conv_spec;
            } else {
                LOG(FATAL) << "Unsupported";
            }
        }

        return std::move(default_graph_specification);
    }

    void TuningSpaceGenerator::set_allow_and_reject_list(
            GraphSpecification *graph_spec,
            Graph *graph,
            std::string allowed_node) {
        for (auto const &node_name : graph->get_node_list()) {
            if (node_name != allowed_node) {
                graph_spec->add_codegen_reject_list(node_name);
            }
        }

        graph_spec->add_codegen_allow_list(allowed_node);
    }
}
}