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

#include "mononn_engine/tuning/graph_tuner.h"

#include <experimental/filesystem>
#include <fstream>
#include <limits>

#include "google/protobuf/util/json_util.h"
#include "mononn_engine/config/config.h"
#include "mononn_engine/helpers/helpers.h"
#include "tensorflow/mononn_extra/proto/cluster_elewise_specification.pb.h"
#include "tensorflow/mononn_extra/proto/cluster_reduce_specification.pb.h"
#include "tensorflow/mononn_extra/proto/cuda_context.pb.h"
#include "tensorflow/mononn_extra/proto/dim3.pb.h"

namespace mononn_engine {
namespace tuning {
namespace fs = std::experimental::filesystem;
using Config = mononn_engine::config::Config;
using GraphSpecification = tensorflow::mononn_extra::proto::GraphSpecification;
namespace proto = tensorflow::mononn_extra::proto;

std::unique_ptr<GraphSpecification> GraphTuner::get_optimal_spec(
    std::vector<GraphSpecification const*> candidate_spec_list) {
  std::vector<std::future<ProfilingResult>> profiling_result_futures;
  std::vector<ProfilingResult> profiling_result;
  LOG(INFO) << "Begin individual profiling";
  // individual profiling
  for (auto spec : candidate_spec_list) {
    profiling_result_futures.push_back(std::move(this->profiling_queue.post(
        spec, {"generate_memory_initialization",
               "generate_parameter_initialization"})));
  }

  int current_task_id = 0;
  LOG(INFO) << "Waiting for task complete...";

  for (auto& future : profiling_result_futures) {
    future.wait();

    current_task_id += 1;

    if (current_task_id % 10 == 0) {
      LOG(INFO) << "Finish " << current_task_id << " out of "
                << candidate_spec_list.size() << " tasks";
    }

    auto result = future.get();

    this->check_profiling_result(result);

    profiling_result.push_back(result);
  }

  // End-to-end profiling
  LOG(INFO) << "Begin end to end profiling";
  std::vector<std::unique_ptr<GraphSpecification>> best_spec_for_each_context =
      std::move(GraphTuner::get_optimal_spec_for_each_cuda_context(
          profiling_result, candidate_spec_list));
  std::vector<std::future<ProfilingResult>> e2e_profiling_result_future;
  std::vector<ProfilingResult> e2e_profiling_result;

  if (Config::get()->save_candidate_specification) {
    std::string candidate_spec_save_path = mononn_engine::helpers::Path::join(
        Config::get()->output_dir, "candidate_specs");
    LOG(INFO) << "Save candidate specification to " << candidate_spec_save_path;

    if (fs::exists(candidate_spec_save_path)) {
      fs::remove_all(candidate_spec_save_path);
    }

    fs::create_directories(candidate_spec_save_path);

    for (auto& spec : best_spec_for_each_context) {
      proto::Dim3 grid_dim =
          spec->cuda_context().cuda_runtime_context().grid_dim();
      proto::Dim3 block_dim =
          spec->cuda_context().cuda_runtime_context().block_dim();

      std::string context_str = mononn_engine::helpers::string_format(
          "%d__%d", grid_dim.x(), block_dim.x());

      std::string save_path = mononn_engine::helpers::Path::join(
          candidate_spec_save_path, context_str);

      fs::create_directories(save_path);

      std::string json_file_name =
          mononn_engine::helpers::Path::join(save_path, "graph_spec.json");
      std::string proto_file_name =
          mononn_engine::helpers::Path::join(save_path, "graph_spec.pb");

      mononn_engine::helpers::save_proto_to_json_file(spec.get(),
                                                      json_file_name);
      mononn_engine::helpers::save_proto_to_binary_file(spec.get(),
                                                        proto_file_name);
    }
  }

  for (auto& spec : best_spec_for_each_context) {
    e2e_profiling_result_future.push_back(std::move(this->profiling_queue.post(
        spec.get(), {"generate_memory_initialization",
                     "generate_parameter_initialization"})));
  }

  for (auto& future : e2e_profiling_result_future) {
    future.wait();

    e2e_profiling_result.push_back(future.get());
  }

  for (auto& result : e2e_profiling_result) {
    this->check_profiling_result(result);
  }

  std::stringstream ss;
  for (int idx = 0; idx < best_spec_for_each_context.size(); ++idx) {
    const proto::CUDAContext& context =
        best_spec_for_each_context[idx]->cuda_context();
    const proto::Dim3 grid_dim = context.cuda_runtime_context().grid_dim();
    const proto::Dim3 block_dim = context.cuda_runtime_context().block_dim();

    std::string context_str = mononn_engine::helpers::string_format(
        "%d_%d_%d__%d_%d_%d", grid_dim.x(), grid_dim.y(), grid_dim.z(),
        block_dim.x(), block_dim.y(), block_dim.z());

    LOG(INFO) << "Context: " << context_str << " best solution time: "
              << e2e_profiling_result[idx].time_in_us;
    ss << "Context: " << context_str
       << " best solution time: " << e2e_profiling_result[idx].time_in_us
       << "\n";
  }

  // write performance log
  std::ofstream ofs(mononn_engine::helpers::Path::join(
      Config::get()->output_dir, "tuning_log.log"));
  ofs << ss.str();
  ofs.close();

  LOG(INFO) << "Choose best solution";

  int best_id = 0;
  for (int idx = 1; idx < e2e_profiling_result.size(); ++idx) {
    if (e2e_profiling_result[idx].time_in_us <
        e2e_profiling_result[best_id].time_in_us) {
      best_id = idx;
    }
  }

  return std::move(best_spec_for_each_context[best_id]);
}

std::vector<std::unique_ptr<GraphSpecification>>
GraphTuner::get_optimal_spec_for_each_cuda_context(
    std::vector<ProfilingResult>& profiling_result,
    std::vector<const GraphSpecification*> candidate_spec_list) {
  std::vector<std::unique_ptr<GraphSpecification>> result;

  std::unordered_map<std::string, std::vector<const GraphSpecification*>>
      spec_by_context;
  std::unordered_map<std::string, std::vector<ProfilingResult>>
      profiling_result_by_context;

  for (int idx = 0; idx < profiling_result.size(); ++idx) {
    const GraphSpecification* spec = candidate_spec_list[idx];

    std::string context_str = GraphTuner::context_to_str(&spec->cuda_context());

    if (spec_by_context.count(context_str) == 0) {
      spec_by_context[context_str] = std::vector<const GraphSpecification*>();
      profiling_result_by_context[context_str] = std::vector<ProfilingResult>();
    }

    spec_by_context[context_str].push_back(spec);
    profiling_result_by_context[context_str].push_back(profiling_result[idx]);
  }

  for (auto& [context_str, spec_list] : spec_by_context) {
    std::unique_ptr<GraphSpecification> best_spec_for_context =
        GraphTuner::get_optimal_spec_for_cuda_context(
            profiling_result_by_context[context_str], spec_list);

    result.push_back(std::move(best_spec_for_context));
  }

  return std::move(result);
}

std::unique_ptr<GraphSpecification>
GraphTuner::get_optimal_spec_for_cuda_context(
    std::vector<ProfilingResult>& profiling_result,
    std::vector<const GraphSpecification*> candidate_spec_list) {
  std::unique_ptr<GraphSpecification> best_spec_for_cuda_context =
      mononn_engine::helpers::deep_copy_graph_specification(
          candidate_spec_list[0]);

  for (auto node_name : best_spec_for_cuda_context->codegen_reject_list()) {
    (*best_spec_for_cuda_context->mutable_codegen_allow_list())
        .Add(std::move(node_name));
  }

  (*best_spec_for_cuda_context->mutable_codegen_reject_list()).Clear();

  std::unordered_map<std::string, std::vector<const GraphSpecification*>>
      spec_by_node;
  std::unordered_map<std::string, std::vector<ProfilingResult>>
      profiling_result_by_node;

  for (int idx = 0; idx < candidate_spec_list.size(); ++idx) {
    std::string node_name = candidate_spec_list[idx]->codegen_allow_list(0);

    if (spec_by_node.count(node_name) == 0) {
      spec_by_node[node_name] = std::vector<const GraphSpecification*>();
      profiling_result_by_node[node_name] = std::vector<ProfilingResult>();
    }

    spec_by_node[node_name].push_back(candidate_spec_list[idx]);
    profiling_result_by_node[node_name].push_back(profiling_result[idx]);
  }

  for (auto& [node_name, spec_list_for_node] : spec_by_node) {
    std::unique_ptr<GraphSpecification> best_node_spec =
        GraphTuner::get_optimal_spec_for_node(
            profiling_result_by_node[node_name], spec_by_node[node_name]);

    if (best_node_spec->gemm_spec_list().contains(node_name)) {
      (*best_spec_for_cuda_context->mutable_gemm_spec_list())[node_name] =
          best_node_spec->gemm_spec_list().at(node_name);
    } else if (best_node_spec->cluster_elewise_spec().contains(node_name)) {
      (*best_spec_for_cuda_context->mutable_cluster_elewise_spec())[node_name] =
          best_node_spec->cluster_elewise_spec().at(node_name);
    } else if (best_node_spec->cluster_reduce_spec().contains(node_name)) {
      (*best_spec_for_cuda_context->mutable_cluster_reduce_spec())[node_name] =
          best_node_spec->cluster_reduce_spec().at(node_name);
    } else if (best_node_spec->conv_spec_list().contains(node_name)) {
      (*best_spec_for_cuda_context->mutable_conv_spec_list())[node_name] =
          best_node_spec->conv_spec_list().at(node_name);
    } else {
      LOG(FATAL) << "Node " << node_name << "not found.";
    }
  }

  return std::move(best_spec_for_cuda_context);
}

std::unique_ptr<GraphSpecification> GraphTuner::get_optimal_spec_for_node(
    std::vector<ProfilingResult>& profiling_result,
    std::vector<const GraphSpecification*> candidate_spec_list) {
  int best_id = 0;
  for (int idx = 1; idx < profiling_result.size(); ++idx) {
    if (profiling_result[idx].time_in_us <
        profiling_result[best_id].time_in_us) {
      best_id = idx;
    }
  }

  return std::move(mononn_engine::helpers::deep_copy_graph_specification(
      candidate_spec_list[best_id]));
}

std::string GraphTuner::context_to_str(
    const tensorflow::mononn_extra::proto::CUDAContext* cuda_context) {
  auto& grid_dim = cuda_context->cuda_runtime_context().grid_dim();
  auto& block_dim = cuda_context->cuda_runtime_context().block_dim();

  std::string result = mononn_engine::helpers::string_format(
      "%d_%d_%d__%d_%d_%d", grid_dim.x(), grid_dim.y(), grid_dim.z(),
      block_dim.x(), block_dim.y(), block_dim.z());

  return result;
}

void GraphTuner::check_profiling_result(const ProfilingResult& result) const {
  if (!result.codegen_success) {
    LOG(FATAL) << "Codegen failed "
               << "\n"
               << result.output << "\n"
               << "Debug directory:" << result.profiling_directory;
  }

  if (!result.build_success) {
    LOG(FATAL) << "Build failed"
               << "\n"
               << result.output << "\n"
               << "Debug directory:" << result.profiling_directory;
  }

  if (!result.profile_success) {
    LOG(FATAL) << "Profile failed"
               << "\n"
               << result.output << "\n"
               << "Debug directory:" << result.profiling_directory;
  }
}
}  // namespace tuning
}  // namespace mononn_engine