#include <fstream>
#include <experimental/filesystem>
#include "mononn_engine/tuning/graph_cached_tuner.h"
#include "mononn_engine/helpers/protobuf.h"
#include "mononn_engine/config/config.h"
#include "mononn_engine/helpers/helpers.h"
#include "google/protobuf/util/message_differencer.h"

namespace mononn_engine {
namespace tuning {
    using GraphSpecification = tensorflow::mononn_extra::proto::GraphSpecification;
    using Config = mononn_engine::config::Config;
    namespace fs = std::experimental::filesystem;
    namespace proto = tensorflow::mononn_extra::proto;

    std::unique_ptr<GraphSpecification>
    GraphCachedTuner::get_optimal_spec(std::vector<const GraphSpecification *> candidate_spec_list) {
        std::unordered_map<std::string,
            std::unordered_map<std::string,
                std::vector<std::pair<const GraphSpecification *, std::future<ProfilingResult>>>>> future_result_list_by_context_by_node;

        std::unordered_map<std::string,
            std::unordered_map<std::string,
                const GraphSpecification *>> optimal_specs_by_context_by_node;
        std::unordered_map<std::string, std::unordered_map<std::string, float>> performance_report; // mapping: context -> node -> time

        LOG(INFO) << "Begin profiling non GEMM nodes";
        int non_gemm_nodes_optimization_count = 0;
        int finished_non_gemm_node_optimization_count = 0;

        for (auto &spec : candidate_spec_list) {
            if (spec->codegen_allow_list_size() != 1) {
                LOG(FATAL) << "Codegen allow list is not one: ";
            }

            std::string context_str = GraphCachedTuner::context_to_str(&spec->cuda_context());
            std::string node_name = spec->codegen_allow_list(0);

            if (this->graph->get_node(node_name)->is_gemm() || this->graph->get_node(node_name)->is_conv()) {
                continue;
            } else {
                non_gemm_nodes_optimization_count++;
            }

            if (future_result_list_by_context_by_node.count(context_str) == 0) {
                future_result_list_by_context_by_node[context_str] = std::unordered_map<std::string, std::vector<std::pair<const GraphSpecification *, std::future<ProfilingResult>>>>();
            }

            if (future_result_list_by_context_by_node[context_str].count(node_name) == 0) {
                future_result_list_by_context_by_node[context_str][node_name] = std::vector<std::pair<const GraphSpecification *, std::future<ProfilingResult>>>();
            }

            future_result_list_by_context_by_node[context_str][node_name].push_back(std::move(std::make_pair(spec, this->profiling_queue.post(spec, {"generate_memory_initialization", "generate_parameter_initialization"}))));
        }

        for (auto &[context_str, future_result_by_node] : future_result_list_by_context_by_node) {
            for (auto &[node_name, spec_list] : future_result_by_node) {
                if (optimal_specs_by_context_by_node.count(context_str) == 0) {
                    optimal_specs_by_context_by_node[context_str] = std::unordered_map<std::string, const GraphSpecification *>();
                }

                float best_time = -1;

                for (auto &[spec, future_result] : spec_list) {
                    future_result.wait();
                    ++finished_non_gemm_node_optimization_count;

                    if (finished_non_gemm_node_optimization_count % 10 == 0) {
                        LOG(INFO) << "Non gemm node profiling progress: " << finished_non_gemm_node_optimization_count << " out of " << non_gemm_nodes_optimization_count;
                    }

                    auto result = future_result.get();
                    this->check_profiling_result(result);

                    if (best_time < 0 || best_time > result.time_in_us) {
                        best_time = result.time_in_us;
                        performance_report[context_str][node_name] = best_time;
                        optimal_specs_by_context_by_node[context_str][node_name] = spec;
                    }
                }
            }
        }

        LOG(INFO) << "End profiling non GEMM nodes";

        LOG(INFO) << "Begin profiling GEMM nodes";
        std::unordered_map<std::string,
            std::unordered_map<std::string,
                std::vector<const GraphSpecification *>>> spec_by_node_by_context;
        std::unordered_map<std::string,
            std::unordered_map<std::string,
                proto::CutlassConfig>> cached_cutlass_config_by_node_by_context;

        for (auto &spec : candidate_spec_list) {
            if (spec->codegen_allow_list_size() != 1) {
                LOG(FATAL) << "Codegen allow list is not one: ";
            }

            std::string node_name = spec->codegen_allow_list(0);

            if (!(this->graph->get_node(node_name)->is_gemm() || this->graph->get_node(node_name)->is_conv())) continue;

            std::string context_str = this->context_to_str(&spec->cuda_context());

            if (spec_by_node_by_context.count(node_name) == 0) {
                spec_by_node_by_context[node_name] = std::unordered_map<std::string, std::vector<const GraphSpecification *>>();
            }

            if (spec_by_node_by_context[node_name].count(context_str) == 0) {
                spec_by_node_by_context[node_name][context_str] = std::vector<const GraphSpecification *>();
            }

            spec_by_node_by_context[node_name][context_str].push_back(spec);
        }

        int gemm_nodes_optimization_count = (int)spec_by_node_by_context.size();

        int finished_gemm_node_optimization_count = 0;

        for (auto &[node_name, spec_by_context] : spec_by_node_by_context) {
            std::string gemm_problem_hash = this->get_gemm_or_conv_problem_hash(this->graph->get_node(node_name).get());
            if (Config::get()->use_cached_tuning_result &&
                cached_cutlass_config_by_node_by_context.count(gemm_problem_hash)) {
                LOG(INFO) << "Node " << node_name << " hit profiling cache, problem hash: " << gemm_problem_hash;
                for (auto &[context_str, spec_list] : spec_by_context) {
                    auto best_spec = std::find_if(spec_list.begin(), spec_list.end(), [&](const GraphSpecification *spec) -> bool {
                        return google::protobuf::util::MessageDifferencer::Equivalent(spec->gemm_spec_list().at(node_name).cutlass_config(),
                                       cached_cutlass_config_by_node_by_context[gemm_problem_hash][context_str]);
                    });

                    if (best_spec == spec_list.end()) { LOG(FATAL) << "Cannot find cached specification. Node " << node_name << " context: " << context_str; }
                    optimal_specs_by_context_by_node[context_str][node_name] = *best_spec;
                }
            } else {
                LOG(INFO) << "Tune GEMM node " << node_name << " problem hash: " << gemm_problem_hash;
                std::unordered_map<std::string,
                    std::vector<std::pair<const GraphSpecification *, std::future<ProfilingResult>>>> gemm_futures_by_context;

                if (!cached_cutlass_config_by_node_by_context.count(gemm_problem_hash)) {
                    cached_cutlass_config_by_node_by_context[gemm_problem_hash] = std::unordered_map<std::string, proto::CutlassConfig>();
                }
                
                for (auto &[context_str, spec_list] : spec_by_context) {
                    gemm_futures_by_context[context_str] =
                            std::vector<std::pair<const GraphSpecification *, std::future<ProfilingResult>>>();
                    for (auto &spec : spec_list) {
                        gemm_futures_by_context[context_str].push_back(std::move(std::make_pair(spec, this->profiling_queue.post(spec, {"generate_memory_initialization", "generate_parameter_initialization"}))));
                    }
                }

                for (auto &[context_str, future_list] : gemm_futures_by_context) {
                    float best_time = -1;

                    for (auto &[spec, future] : future_list) {
                        future.wait();
                        auto result = future.get();
                        this->check_profiling_result(result);

                        if (best_time < 0 || best_time > result.time_in_us) {
                            best_time = result.time_in_us;
                            
                            performance_report[context_str][node_name] = best_time;
                            optimal_specs_by_context_by_node[context_str][node_name] = spec;
                        }
                    }

                    cached_cutlass_config_by_node_by_context[gemm_problem_hash][context_str] =
                            optimal_specs_by_context_by_node[context_str][node_name]->gemm_spec_list().at(node_name).cutlass_config();
                }
            }

            ++finished_gemm_node_optimization_count;
            if (finished_gemm_node_optimization_count % 10 == 0) {
                LOG(INFO) << "Gemm node profiling progess: " << finished_gemm_node_optimization_count << " out of " << gemm_nodes_optimization_count;
            }
        }

        LOG(INFO) << "End profiling GEMM nodes";

        LOG(INFO) << "Begin final profiling";
        std::unordered_map<std::string, std::unique_ptr<GraphSpecification>> optimal_spec_for_each_context;
        std::unordered_map<std::string, std::future<ProfilingResult>> future_result_for_each_context;
        std::unordered_map<std::string, ProfilingResult> result_for_each_context;

        this->dump_performance_report(performance_report);

        for (auto &[context_str, specs_by_node] : optimal_specs_by_context_by_node) {
            std::unique_ptr<GraphSpecification> optimal_spec_for_context = mononn_engine::helpers::deep_copy_graph_specification(specs_by_node.begin()->second);
            for (auto node_name : optimal_spec_for_context->codegen_reject_list()) {
                optimal_spec_for_context->mutable_codegen_allow_list()->Add(std::move(node_name));
            }

            optimal_spec_for_context->mutable_codegen_reject_list()->Clear();

            for (auto &[node_name, spec] : specs_by_node) {
                if (optimal_spec_for_context->gemm_spec_list().contains(node_name)) {
                    (*optimal_spec_for_context->mutable_gemm_spec_list())[node_name] = spec->gemm_spec_list().at(node_name);
                } else if (optimal_spec_for_context->conv_spec_list().contains(node_name)) {
                    (*optimal_spec_for_context->mutable_conv_spec_list())[node_name] = spec->conv_spec_list().at(node_name);
                } else if (optimal_spec_for_context->cluster_elewise_spec().contains(node_name)) {
                    (*optimal_spec_for_context->mutable_cluster_elewise_spec())[node_name] = spec->cluster_elewise_spec().at(node_name);
                } else if (optimal_spec_for_context->cluster_reduce_spec().contains(node_name)) {
                    (*optimal_spec_for_context->mutable_cluster_reduce_spec())[node_name] = spec->cluster_reduce_spec().at(node_name);
                } else {
                    LOG(FATAL) << "Node " << node_name << "not found.";
                }
            }

            future_result_for_each_context[context_str] = std::move(this->profiling_queue.post(optimal_spec_for_context.get()));
            optimal_spec_for_each_context[context_str] = std::move(optimal_spec_for_context);
        }

        std::string hlo_module_proto_file = mononn_engine::helpers::Path::join(Config::get()->output_dir, "hlo_module.pb");

        if (Config::get()->save_candidate_specification) {
            std::string candidate_spec_save_path = mononn_engine::helpers::Path::join(Config::get()->output_dir, "candidate_specs");
            LOG(INFO) << "Save candidate specification to " << candidate_spec_save_path;

            if (fs::exists(candidate_spec_save_path)) {
                fs::remove_all(candidate_spec_save_path);
            }

            fs::create_directories(candidate_spec_save_path);

            for (auto &[context_str, spec] : optimal_spec_for_each_context) {
                auto spec_to_save = mononn_engine::helpers::deep_copy_graph_specification(spec.get());
                spec_to_save->set_hlo_module_proto_file(hlo_module_proto_file);
                std::string save_path = mononn_engine::helpers::Path::join(candidate_spec_save_path, context_str);

                fs::create_directories(save_path);

                std::string json_file_name = mononn_engine::helpers::Path::join(save_path, "graph_spec.json");
                std::string proto_file_name = mononn_engine::helpers::Path::join(save_path, "graph_spec.pb");

                mononn_engine::helpers::save_proto_to_json_file(spec_to_save.get(), json_file_name);
                mononn_engine::helpers::save_proto_to_binary_file(spec_to_save.get(), proto_file_name);
            }
        }

        for (auto &[context_str, future] : future_result_for_each_context) {
            future.wait();

            auto result = future.get();
            this->check_profiling_result(result);

            result_for_each_context[context_str] = result;
        }

        float best_result = -1;
        std::stringstream ss;
        std::unique_ptr<GraphSpecification> final_result;
        for (auto &[context_str, result] : result_for_each_context) {
            LOG(INFO) << "Context: " << context_str << " best solution time: " << result.time_in_us;
            ss << "Context: " << context_str << " best solution time: " <<  result.time_in_us << "\n";

            if (best_result < 0 || best_result > result.time_in_us) {
                best_result = result.time_in_us;
                final_result = std::move(optimal_spec_for_each_context[context_str]);
            }
        }

        std::ofstream ofs(mononn_engine::helpers::Path::join(Config::get()->output_dir, "tuning_log.log"));
        ofs << ss.str();
        ofs.close();

        // save hlo module proto
        LOG(INFO) << "Copy " << Config::get()->hlo_module_proto_temp_file << " to " << hlo_module_proto_file;
        final_result->set_hlo_module_proto_file(hlo_module_proto_file);

        if (mononn_engine::helpers::File::exists(hlo_module_proto_file)) {
            mononn_engine::helpers::File::remove(hlo_module_proto_file);
        }

        mononn_engine::helpers::File::copy(Config::get()->hlo_module_proto_temp_file, hlo_module_proto_file);

        return std::move(final_result);
    }

    std::string GraphCachedTuner::context_to_str(const tensorflow::mononn_extra::proto::CUDAContext *cuda_context) {
        auto &grid_dim = cuda_context->cuda_runtime_context().grid_dim();
        auto &block_dim = cuda_context->cuda_runtime_context().block_dim();

        std::string result = mononn_engine::helpers::string_format("%d_%d_%d__%d_%d_%d",
                  grid_dim.x(), grid_dim.y(), grid_dim.z(), block_dim.x(), block_dim.y(), block_dim.z());

        return result;
    }

    std::string GraphCachedTuner::get_gemm_or_conv_problem_hash(const Op *op) const {
        std::string problem_hash;

        if (op->is_gemm()) {
            problem_hash = "_gemm_";
        } else if (op->is_conv()) {
            problem_hash = "_conv_";
        } else {
            LOG(FATAL) << "Invalid node: " << op->get_name();    
        }

        problem_hash += op->get_operand(0)->get_output_spec(0).get_dtype().to_string();
        problem_hash += op->get_operand(0)->get_output_spec(0).to_string();

        problem_hash += op->get_operand(1)->get_output_spec(0).get_dtype().to_string();
        problem_hash += op->get_operand(1)->get_output_spec(0).to_string();

        problem_hash += op->get_output_spec(0).to_string();
        problem_hash += op->get_output_spec(0).to_string();
        return problem_hash;
    }

    void GraphCachedTuner::check_profiling_result(const ProfilingResult &result) const {
        if (!result.codegen_success) {
            LOG(FATAL) << "Codegen failed " << "\n" << result.output
                       << "\n" << "Debug directory:" << result.profiling_directory;
        }

        if (!result.build_success) {
            LOG(FATAL) << "Build failed" << "\n" << result.output
                       << "\n" << "Debug directory:" << result.profiling_directory;
        }

        if (!result.profile_success) {
            LOG(FATAL) << "Profile failed" << "\n" << result.output
                       << "\n" << "Debug directory:" << result.profiling_directory;
        }
    }

    void GraphCachedTuner::dump_performance_report(const std::unordered_map<std::string, std::unordered_map<std::string, float>> &report) const {
        std::stringstream ss;

        for (auto const &[context_str, time_by_node] : report) {
            for (auto const &[node_name, time] : time_by_node) {
                ss << context_str << "," << node_name << "," << time << "\n";
            }
        }

        std::ofstream ofs(mononn_engine::helpers::Path::join(Config::get()->output_dir, "performance_report.csv"));
        ofs << ss.str();
        ofs.close();
    }
}
}