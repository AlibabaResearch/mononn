#include <experimental/filesystem>
#include <fstream>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mononn_engine/codegen/graph_specification_codegen.h"
#include "mononn_engine/config/config.h"
#include "mononn_engine/helpers/helpers.h"
#include "mononn_engine/optimization/clustering_single_node_pass.h"
#include "mononn_engine/optimization/element_wise_concatenation_pass.h"
#include "mononn_engine/optimization/global_synchronization_assignment_pass.h"
#include "mononn_engine/optimization/global_synchronization_elimination_pass.h"
#include "mononn_engine/optimization/intra_op_reschedule_pass.h"
#include "mononn_engine/optimization/merge_dependent_pass.h"
#include "mononn_engine/optimization/merge_independent_pass.h"
#include "mononn_engine/optimization/one_time_pass_runner.h"
#include "mononn_engine/optimization/optimization_runner.h"
#include "mononn_engine/optimization/pass_manager.h"
#include "mononn_engine/optimization/pass_runner.h"
#include "mononn_engine/optimization/run_greedy_pass_runner.h"
#include "mononn_engine/optimization/vectorization_pass.h"
#include "mononn_engine/parser/hlo_module_dumper.h"
#include "mononn_engine/parser/ir_parser_fused.h"
#include "mononn_engine/tuning/graph_cached_tuner.h"
#include "mononn_engine/tuning/graph_tuner.h"
#include "mononn_engine/tuning/options.h"
#include "mononn_engine/tuning/profiler/subprocess.h"
#include "mononn_engine/tuning/tuning_space_generator.h"

ABSL_FLAG(std::string, input_file, "", "The input frozen pb file");
ABSL_FLAG(int, num_threads, 20, "The number of subprocess for parallel tuning");
ABSL_FLAG(std::string, output_dir, "", "The directory for tuned model.");
ABSL_FLAG(std::string, dump_text_hlo_dir, "", "The directory for text hlo");
ABSL_FLAG(std::vector<std::string>, gpus, {}, "Gpu list for profiling");
ABSL_FLAG(bool, automatic_mixed_precision, false,
          "Weather tensor core is enabled");
ABSL_FLAG(bool, faster_tuning, false,
          "Prune tuning space for faster tuning, may get sub-optimal solution");
ABSL_FLAG(bool, fastest_tuning, false,
          "Minimize tuning space, result in slow solution, this is for fast "
          "prototyping");
ABSL_FLAG(bool, use_cached_tuning_result, true,
          "Use tuning cache to expedite tuning");
ABSL_FLAG(std::vector<std::string>, feeds, {}, "Graph input nodes");
ABSL_FLAG(std::vector<std::string>, input_data_files, {}, "Input data files");
ABSL_FLAG(std::vector<std::string>, fetches, {}, "Graph output nodes");
ABSL_FLAG(std::vector<std::string>, optimization_disabled_pass, {},
          "Pass disabled during optimization");

using TuningSpaceGenerator = mononn_engine::tuning::TuningSpaceGenerator;
using GraphTuner = mononn_engine::tuning::GraphTuner;
using GraphCachedTuner = mononn_engine::tuning::GraphCachedTuner;
using GraphSpecificationCodegen =
    mononn_engine::codegen::GraphSpecificationCodegen;
using CUDAProgram = mononn_engine::codegen::CUDAProgram;
using Config = mononn_engine::config::Config;
using Graph = mononn_engine::core::graph::Graph;
using PassManager = mononn_engine::optimization::PassManager;
using PassRunner = mononn_engine::optimization::PassRunner;
using OneTimePassRunner = mononn_engine::optimization::OneTimePassRunner;
using RunGreedyPassRunner = mononn_engine::optimization::RunGreedyPassRunner;
using MergeDependentPass = mononn_engine::optimization::MergeDependentPass;
using MergeIndependentPass = mononn_engine::optimization::MergeIndependentPass;
using IntraOpReschedulePass =
    mononn_engine::optimization::IntraOpReschedulePass;
using GlobalSynchronizationEliminationPass =
    mononn_engine::optimization::GlobalSynchronizationEliminationPass;
using GlobalSynchronizationAssignmentPass =
    mononn_engine::optimization::GlobalSynchronizationAssignmentPass;
using ClusteringSingleNodePass =
    mononn_engine::optimization::ClusteringSingleNodePass;
using ElementWiseConcatenationPass =
    mononn_engine::optimization::ElementWiseConcatenationPass;
using HloModuleDumper = mononn_engine::parser::HloModuleDumper;
using VectorizationPass = mononn_engine::optimization::VectorizationPass;
using SubProcess = mononn_engine::tuning::profiler::SubProcess;
using OptimizationRunner = mononn_engine::optimization::OptimizationRunner;

#define ADD_PASS_TO_PASS_MANAGER(pass_manager, pass_runner, pass_name, ...) \
  {                                                                         \
    std::unique_ptr<pass_name> __pass_##pass_name =                         \
        std::make_unique<pass_name>(__VA_ARGS__);                           \
    std::unique_ptr<PassRunner> __pass_runner_##pass_name =                 \
        std::make_unique<pass_runner>(std::move(__pass_##pass_name));       \
    pass_manager.add_runner(std::move(__pass_runner_##pass_name));          \
  }

std::shared_ptr<Graph> get_network_graph(std::string hlo_module_proto_file) {
  std::shared_ptr<Graph> graph =
      mononn_engine::parser::IRParserFused::from_hlo_module_proto_file(
          hlo_module_proto_file);

  OptimizationRunner::run_group_pre_impl_assignment(graph.get(), nullptr);

  // PassManager pm_before_impl(nullptr);
  // ADD_PASS_TO_PASS_MANAGER(pm_before_impl, OneTimePassRunner,
  // ClusteringSingleNodePass); ADD_PASS_TO_PASS_MANAGER(pm_before_impl,
  // RunGreedyPassRunner, ElementWiseConcatenationPass);
  // ADD_PASS_TO_PASS_MANAGER(pm_before_impl, RunGreedyPassRunner,
  // MergeIndependentPass); ADD_PASS_TO_PASS_MANAGER(pm_before_impl,
  // RunGreedyPassRunner, MergeDependentPass);
  // ADD_PASS_TO_PASS_MANAGER(pm_before_impl, OneTimePassRunner,
  // GlobalSynchronizationAssignmentPass);
  // ADD_PASS_TO_PASS_MANAGER(pm_before_impl, OneTimePassRunner,
  // GlobalSynchronizationEliminationPass); pm_before_impl.execute(graph);

  // PassManager pm_vectorization(nullptr);
  // ADD_PASS_TO_PASS_MANAGER(pm_vectorization, OneTimePassRunner,
  // VectorizationPass); pm_vectorization.execute(graph);

  return graph;
}

int Main(mononn_engine::tuning::Options const& options) {
  Config::get()->output_dir = options.output_dir;
  Config::get()->gpus = options.gpus;
  Config::get()->gemm_tensor_op_enabled = options.automatic_mixed_precision;
  Config::get()->hlo_module_proto_temp_path =
      mononn_engine::helpers::Directory::get_mononn_new_temp_dir();
  Config::get()->hlo_module_proto_temp_file =
      mononn_engine::helpers::Path::join(
          Config::get()->hlo_module_proto_temp_path, "hlo_module.pb");
  Config::get()->dump_text_hlo_dir = options.dump_text_hlo_dir;
  Config::get()->feeds = options.feeds;
  Config::get()->fetches = options.fetches;
  Config::get()->input_data_files = options.input_data_files;
  Config::get()->faster_tuning = options.faster_tuning;
  Config::get()->fastest_tuning = options.fastest_tuning;
  Config::get()->use_cached_tuning_result = options.use_cached_tuning_result;
  Config::get()->optimization_disabled_pass =
      options.optimization_disabled_pass;

  mononn_engine::helpers::Directory::create_recursive(
      Config::get()->hlo_module_proto_temp_path);
  HloModuleDumper hlo_module_dumper(
      options.input_file, options.automatic_mixed_precision, options.feeds,
      options.input_data_files, options.fetches);
  hlo_module_dumper.dump(Config::get()->hlo_module_proto_temp_path,
                         options.dump_text_hlo_dir);

  std::shared_ptr<Graph> graph =
      get_network_graph(Config::get()->hlo_module_proto_temp_file);

  std::vector<
      std::unique_ptr<tensorflow::mononn_extra::proto::GraphSpecification>>
      graph_spec_list = std::move(TuningSpaceGenerator::generate_tuning_space(
          graph.get(), Config::get()->hlo_module_proto_temp_file, options.feeds,
          options.input_data_files, options.fetches));

  LOG(INFO) << "Generate " << graph_spec_list.size()
            << " candidate graph specification";

  std::vector<tensorflow::mononn_extra::proto::GraphSpecification const*>
      graph_spec_list_ptr;

  GraphCachedTuner graph_tuner(options.num_threads, graph);
  std::unique_ptr<tensorflow::mononn_extra::proto::GraphSpecification>
      optimal_spec = graph_tuner.get_optimal_spec(
          mononn_engine::helpers::Transform::map<
              std::unique_ptr<
                  tensorflow::mononn_extra::proto::GraphSpecification>,
              tensorflow::mononn_extra::proto::GraphSpecification const*>(
              graph_spec_list,
              [&](std::unique_ptr<
                  tensorflow::mononn_extra::proto::GraphSpecification>& spec)
                  -> tensorflow::mononn_extra::proto::
                      GraphSpecification const* { return spec.get(); }));

  LOG(INFO) << "Generate final program";
  std::unique_ptr<CUDAProgram> cuda_program =
      GraphSpecificationCodegen::generate(optimal_spec.get());
  LOG(INFO) << "Save generated program to " << options.output_dir;
  cuda_program->generate(options.output_dir);

  LOG(INFO) << "Build generated program";
  std::string build_cmd = "cd " + options.output_dir + " && make";

  SubProcess build_process(build_cmd, {"2>&1"});
  build_process.start();
  build_process.wait();

  if (build_process.get_return_code() != 0) {
    LOG(FATAL) << "Generated program build failed: "
               << build_process.get_output();
  }

  mononn_engine::helpers::save_proto_to_json_file(
      optimal_spec.get(), mononn_engine::helpers::Path::join(
                              options.output_dir, "graph_spec.json"));
  mononn_engine::helpers::save_proto_to_binary_file(
      optimal_spec.get(),
      mononn_engine::helpers::Path::join(options.output_dir, "graph_spec.pb"));

  mononn_engine::helpers::Directory::remove(
      Config::get()->hlo_module_proto_temp_path);
  return 0;
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  mononn_engine::tuning::Options options;

  options.input_file = absl::GetFlag(FLAGS_input_file);
  options.num_threads = absl::GetFlag(FLAGS_num_threads);
  options.output_dir = absl::GetFlag(FLAGS_output_dir);
  options.dump_text_hlo_dir = absl::GetFlag(FLAGS_dump_text_hlo_dir);
  options.input_data_files = absl::GetFlag(FLAGS_input_data_files);
  options.automatic_mixed_precision =
      absl::GetFlag(FLAGS_automatic_mixed_precision);
  options.faster_tuning = absl::GetFlag(FLAGS_faster_tuning);
  options.fastest_tuning = absl::GetFlag(FLAGS_fastest_tuning);
  options.use_cached_tuning_result =
      absl::GetFlag(FLAGS_use_cached_tuning_result);

  for (auto const& gpu : absl::GetFlag(FLAGS_gpus)) {
    options.gpus.push_back(std::stoi(gpu));
  }

  options.feeds = absl::GetFlag(FLAGS_feeds);
  options.fetches = absl::GetFlag(FLAGS_fetches);
  options.optimization_disabled_pass =
      absl::GetFlag(FLAGS_optimization_disabled_pass);

  if (options.input_file == "") {
    LOG(FATAL) << "Please specify input model file";
  }

  if (options.output_dir == "") {
    LOG(FATAL) << "Please specify output directory";
  }

  if (options.input_data_files.empty()) {
    LOG(FATAL) << "Please specify input data files";
  }

  if (options.feeds.empty()) {
    LOG(FATAL) << "Please specify input nodes";
  }

  if (options.fetches.empty()) {
    LOG(FATAL) << "Please specify output nodes";
  }

  if (options.gpus.empty()) {
    LOG(FATAL) << "Please specify available gpu list";
  }

  if (options.feeds.size() != options.input_data_files.size()) {
    LOG(FATAL) << "Input file not match";
  }

  return Main(options);
}