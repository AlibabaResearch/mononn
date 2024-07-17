// Copyright 2023 The MonoNN Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <experimental/filesystem>
#include <fstream>
#include <future>
#include <sstream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mononn_engine/config/config.h"
#include "mononn_engine/helpers/directory.h"
#include "mononn_engine/helpers/file.h"
#include "mononn_engine/helpers/macros.h"
#include "mononn_engine/helpers/protobuf.h"
#include "mononn_engine/tuning/profiler/parallel_profiling_queue.h"
#include "tensorflow/mononn_extra/proto/graph_specification.pb.h"

ABSL_FLAG(std::string, graph_spec_file, "", "Graph specification file");
ABSL_FLAG(std::string, format, "pb", "Input graph spec format");
ABSL_FLAG(std::string, output_file, "", "Output file");
ABSL_FLAG(int, num_threads, 10, "Num threads");
ABSL_FLAG(std::vector<std::string>, gpus, {"0"}, "Gpu list for profiling");
ABSL_FLAG(std::vector<std::string>, optimization_disabled_pass, {}, "");

struct Options {
  std::string graph_spec_file;
  std::string format;
  std::string output_file;
  int num_threads;
  std::vector<int> gpus;
  std::vector<std::string> optimization_disabled_pass;
};

using GraphSpecification = tensorflow::mononn_extra::proto::GraphSpecification;
using ParallelProfilingQueue =
    mononn_engine::tuning::profiler::ParallelProfilingQueue;
using Config = mononn_engine::config::Config;
using ProfilingResult = mononn_engine::tuning::profiler::ProfilingResult;

void dump_perf_result(
    std::unordered_map<std::string, ProfilingResult>& perf_result,
    std::string output_file) {
  std::stringstream ss;

  ss << "Node Name,Time(us)"
     << "\n";

  for (auto const& [node_name, result] : perf_result) {
    ss << node_name << "," << result.time_in_us << "\n";
  }

  std::ofstream ofs(output_file);
  ofs << ss.str();
  ofs.close();
}

int Main(const Options& options) {
  Config::get()->gpus = options.gpus;
  Config::get()->hlo_module_proto_temp_path =
      mononn_engine::helpers::Directory::get_mononn_new_temp_dir();
  mononn_engine::helpers::Directory::create_recursive(
      Config::get()->hlo_module_proto_temp_path);

  std::unique_ptr<GraphSpecification> graph_spec =
      std::make_unique<GraphSpecification>();

  if (options.format == "pb") {
    mononn_engine::helpers::load_proto_from_binary_file(
        graph_spec.get(), options.graph_spec_file);
  } else if (options.format == "json") {
    mononn_engine::helpers::load_proto_from_json_file(graph_spec.get(),
                                                      options.graph_spec_file);
  } else {
    LOG(FATAL) << "Unsupported format " << options.format;
  }

  ParallelProfilingQueue profiling_queue(options.num_threads);
  std::unordered_map<std::string, std::future<ProfilingResult>>
      perf_result_futures;
  std::unordered_map<std::string, ProfilingResult> perf_result;

  std::vector<std::unique_ptr<GraphSpecification>> specs;

  for (auto node_name : graph_spec->codegen_allow_list()) {
    if (!graph_spec->gemm_spec_list().contains(node_name) &&
        !graph_spec->conv_spec_list().contains(node_name) &&
        !graph_spec->cluster_elewise_spec().contains(node_name) &&
        !graph_spec->cluster_reduce_spec().contains(node_name)) {
      continue;
    }

    auto spec = std::move(mononn_engine::helpers::deep_copy_graph_specification(
        graph_spec.get()));

    spec->mutable_codegen_reject_list()->Clear();
    spec->mutable_codegen_allow_list()->Clear();

    *spec->add_codegen_allow_list() = node_name;

    for (auto no_codegen_node_name : graph_spec->codegen_allow_list()) {
      if (no_codegen_node_name != node_name) {
        *spec->add_codegen_reject_list() = no_codegen_node_name;
      }
    }

    specs.push_back(std::move(spec));
  }

  for (auto& spec : specs) {
    std::string node_name = spec->codegen_allow_list().at(0);

    perf_result_futures[node_name] = std::move(profiling_queue.post(
        spec.get(),
        {"generate_memory_initialization", "generate_parameter_initialization"},
        options.optimization_disabled_pass));
  }

  for (auto& spec : specs) {
    std::string node_name = spec->codegen_allow_list().at(0);
    auto& result_future = perf_result_futures[node_name];
    result_future.wait();

    LOG(INFO) << "Get perf data for node " << node_name;

    auto result = result_future.get();

    result.verify();

    perf_result[node_name] = result;
  }

  dump_perf_result(perf_result, options.output_file);

  return 0;
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  Options options;

  options.graph_spec_file = absl::GetFlag(FLAGS_graph_spec_file);
  options.output_file = absl::GetFlag(FLAGS_output_file);
  options.format = absl::GetFlag(FLAGS_format);
  options.num_threads = absl::GetFlag(FLAGS_num_threads);
  options.optimization_disabled_pass =
      absl::GetFlag(FLAGS_optimization_disabled_pass);

  for (auto const& gpu : absl::GetFlag(FLAGS_gpus)) {
    options.gpus.push_back(std::stoi(gpu));
  }

  if (options.graph_spec_file.empty() ||
      !mononn_engine::helpers::File::exists(options.graph_spec_file)) {
    LOG(FATAL) << "Invalid graph_spec_file " << options.graph_spec_file;
  }

  if (options.output_file.empty()) {
    LOG(FATAL) << "Invalid output file " << options.output_file;
  }

  return Main(options);
}