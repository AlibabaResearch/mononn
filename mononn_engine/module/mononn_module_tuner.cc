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

#include "mononn_engine/module/mononn_module_tuner.h"

#include <unordered_map>

#include "google/protobuf/util/message_differencer.h"
#include "mononn_engine/codegen/compilation_threadpool.h"
#include "mononn_engine/core/common/concurrent_queue.h"
#include "mononn_engine/helpers/directory.h"
#include "mononn_engine/helpers/env_variable.h"
#include "mononn_engine/helpers/file.h"
#include "mononn_engine/helpers/protobuf.h"
#include "mononn_engine/optimization/optimization_runner.h"
#include "mononn_engine/parser/ir_parser_fused.h"
#include "mononn_engine/tuning/profiler/cupti_profiling_session.h"
#include "mononn_engine/tuning/profiler/timer.h"
#include "mononn_engine/tuning/tuning_space_generator.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/mononn_extra/proto/cutlass_config.pb.h"
#include "tensorflow/stream_executor/cuda/cuda_driver.h"
#include "tensorflow/stream_executor/gpu/gpu_kernel.h"

namespace mononn_engine {
namespace module {

using Graph = mononn_engine::core::graph::Graph;
using CUDAContext = mononn_engine::core::context::CUDAContext;
using CUDAContextProto = tensorflow::mononn_extra::proto::CUDAContext;
using OptimizationRunner = mononn_engine::optimization::OptimizationRunner;
using TuningSpaceGenerator = mononn_engine::tuning::TuningSpaceGenerator;
using IRParserFused = mononn_engine::parser::IRParserFused;
using GpuDriver = stream_executor::cuda::CUDADriver;
using CompilationThreadpool = mononn_engine::codegen::CompilationThreadpool;
using CuptiProfilingSession =
    mononn_engine::tuning::profiler::CuptiProfilingSession;
using Op = mononn_engine::core::op::Op;
using CutlassConfigProto = tensorflow::mononn_extra::proto::CutlassConfig;

using TimerRAII =
    mononn_engine::tuning::profiler::TimerRAII<std::chrono::milliseconds>;

using TuningSpecId = uint64_t;

template <typename T>
using ConcurrentQueue = mononn_engine::core::common::ConcurrentQueue<T>;

template <int n>
static std::unique_ptr<stream_executor::KernelArgsArrayBase> MakeKernelArgs(
    absl::Span<const stream_executor::DeviceMemoryBase> args, int smem_bytes) {
  auto kernel_args = absl::make_unique<stream_executor::KernelArgsArray<n>>();
  for (const stream_executor::DeviceMemoryBase& buf : args) {
    kernel_args->add_device_memory_argument(buf);
  }

  kernel_args->add_shared_bytes(smem_bytes);

  return kernel_args;
}

CutlassConfigProto extract_cutlass_config_proto(
    const GraphSpecification* graph_spec) {
  if (graph_spec->codegen_allow_list_size() != 1) {
    LOG(FATAL) << "Codegen allow list size: "
               << graph_spec->codegen_allow_list_size();
  }

  auto const& node_name = graph_spec->codegen_allow_list().at(0);

  if (graph_spec->gemm_spec_list().contains(node_name)) {
    return graph_spec->gemm_spec_list().at(node_name).cutlass_config();
  }

  if (graph_spec->conv_spec_list().contains(node_name)) {
    return graph_spec->conv_spec_list().at(node_name).cutlass_config();
  }

  LOG(FATAL) << node_name << " not found.";
}

CuptiProfilingSession::ProfilingResult profile_kernel(
    const MonoNNModuleTuner::Params& params,
    const CUDAContextProto& cuda_context_proto, const std::string& ptx,
    const std::vector<uint8_t>& cubin) {
  TimerRAII timer_profile_kernel("profile_kernel latency: ");

  CuptiProfilingSession profiling_session(
      {CuptiProfilingSession::Metrics::gpu__time_duration_sum}, 1);

  std::unique_ptr<stream_executor::KernelBase> kernel =
      xla::gpu::CreateKernel(params.kernel_name,
                             params.execution_parameters.size(), ptx, cubin,
                             params.stream->parent())
          .ValueOrDie();

  // Set dynamic shared memory size.
  GpuDriver::FuncSetAttribute(
      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
      stream_executor::gpu::AsGpuKernel(kernel.get())->AsGpuFunctionHandle(),
      cuda_context_proto.cuda_runtime_context().smem_size());

  std::unique_ptr<stream_executor::KernelArgsArrayBase> kernel_args;

  static constexpr int kKernelArgsLimit = 1024;

  if (params.execution_parameters.size() <= 64) {
    kernel_args = MakeKernelArgs<64>(
        params.execution_parameters,
        cuda_context_proto.cuda_runtime_context().smem_size());
  } else if (params.execution_parameters.size() <= 256) {
    kernel_args = MakeKernelArgs<256>(
        params.execution_parameters,
        cuda_context_proto.cuda_runtime_context().smem_size());
  } else {
    kernel_args = MakeKernelArgs<kKernelArgsLimit>(
        params.execution_parameters,
        cuda_context_proto.cuda_runtime_context().smem_size());
  }

  return profiling_session.profiling_context([&params, &cuda_context_proto,
                                              &kernel, &kernel_args] {
    auto const& block_dim =
        cuda_context_proto.cuda_runtime_context().block_dim();
    auto const& grid_dim = cuda_context_proto.cuda_runtime_context().grid_dim();

    tensorflow::Status launch_ok = params.stream->parent()->Launch(
        params.stream,
        stream_executor::ThreadDim(block_dim.x(), block_dim.y(), block_dim.z()),
        stream_executor::BlockDim(grid_dim.x(), grid_dim.y(), grid_dim.z()),
        *kernel, *kernel_args);

    if (!launch_ok.ok()) {
      LOG(FATAL) << "Kernel launch failure, " << launch_ok.error_message();
    }

    tensorflow::Status sync_status = params.stream->BlockHostUntilDone();

    if (!sync_status.ok()) {
      LOG(FATAL) << "Stream synchronize error, " << sync_status.error_message();
    }
  });
}

// Do not really use this at this moment as TF executor still lock on mutex when
// doing PTX jit. May need parallelize PTX jit in command line codegen phase.
CuptiProfilingSession::ProfilingResult profile_kernel_in_batch(
    const MonoNNModuleTuner::Params& params,
    const CUDAContextProto& cuda_context_proto,
    const std::vector<std::string>& ptx_list,
    const std::vector<std::vector<uint8_t>>& cubin_list,
    CompilationThreadpool& compilation_threadpool) {
  TimerRAII timer_profile_kernel("profile_kernel_in_batch latency: ");

  if (ptx_list.size() != cubin_list.size()) {
    LOG(DEBUG) << "Size does not match " << ptx_list.size() << " vs "
               << cubin_list.size();
  }

  int work_count = ptx_list.size();

  CuptiProfilingSession profiling_session(
      {CuptiProfilingSession::Metrics::gpu__time_duration_sum}, work_count);
  std::vector<std::future<std::unique_ptr<stream_executor::KernelBase>>>
      kenrel_future_list;
  std::vector<std::unique_ptr<stream_executor::KernelBase>> kernel_list;

  for (int idx = 0; idx < work_count; ++idx) {
    kenrel_future_list.push_back(std::move(compilation_threadpool.post_general(
        [&ptx_list, &cubin_list, idx,
         &params]() -> std::unique_ptr<stream_executor::KernelBase> {
          std::unique_ptr<stream_executor::KernelBase> kernel =
              xla::gpu::CreateKernel(
                  params.kernel_name, params.execution_parameters.size(),
                  ptx_list[idx], cubin_list[idx], params.stream->parent())
                  .ValueOrDie();

          return std::move(kernel);
        })));
  }

  for (auto& kernel_future : kenrel_future_list) {
    kernel_future.wait();
    kernel_list.push_back(std::move(kernel_future.get()));
  }

  std::unique_ptr<stream_executor::KernelArgsArrayBase> kernel_args;
  std::vector<CuptiProfilingSession::ProfilingResult> profiling_result_list;

  static constexpr int kKernelArgsLimit = 1024;

  if (params.execution_parameters.size() <= 64) {
    kernel_args = MakeKernelArgs<64>(
        params.execution_parameters,
        cuda_context_proto.cuda_runtime_context().smem_size());
  } else if (params.execution_parameters.size() <= 256) {
    kernel_args = MakeKernelArgs<256>(
        params.execution_parameters,
        cuda_context_proto.cuda_runtime_context().smem_size());
  } else {
    kernel_args = MakeKernelArgs<kKernelArgsLimit>(
        params.execution_parameters,
        cuda_context_proto.cuda_runtime_context().smem_size());
  }

  return profiling_session.profiling_context([&params, &cuda_context_proto,
                                              &kernel_args, &kernel_list] {
    for (auto& kernel : kernel_list) {
      GpuDriver::FuncSetAttribute(
          CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
          stream_executor::gpu::AsGpuKernel(kernel.get())
              ->AsGpuFunctionHandle(),
          cuda_context_proto.cuda_runtime_context().smem_size());

      auto const& block_dim =
          cuda_context_proto.cuda_runtime_context().block_dim();
      auto const& grid_dim =
          cuda_context_proto.cuda_runtime_context().grid_dim();

      tensorflow::Status launch_ok = params.stream->parent()->Launch(
          params.stream,
          stream_executor::ThreadDim(block_dim.x(), block_dim.y(),
                                     block_dim.z()),
          stream_executor::BlockDim(grid_dim.x(), grid_dim.y(), grid_dim.z()),
          *kernel, *kernel_args);

      if (!launch_ok.ok()) {
        LOG(FATAL) << "Kernel launch failure, " << launch_ok.error_message();
      }

      tensorflow::Status sync_status = params.stream->BlockHostUntilDone();

      if (!sync_status.ok()) {
        LOG(FATAL) << "Stream synchronize error, "
                   << sync_status.error_message();
      }
    }
  });
}

std::vector<std::tuple<GraphSpecification*, std::string, double>>
profile_in_batch(ConcurrentQueue<CompilationThreadpool::Result>& finish_queue,
                 int work_count, const MonoNNModuleTuner::Params& params) {
  TimerRAII timer_profile_kernel("profile_in_batch latency: ");
  CuptiProfilingSession profiling_session(
      {CuptiProfilingSession::Metrics::gpu__time_duration_sum}, work_count);

  std::vector<std::tuple<GraphSpecification*, std::string, double>> result_list(
      work_count);
  std::vector<CompilationThreadpool::Result::FileCollection>
      file_collection_list(work_count);

  auto error_handle_and_crash = [&](tensorflow::Status& err_status,
                                    const std::string& ptx, int idx) {
    LOG(ERROR) << err_status.error_message();
    std::string debug_dir =
        mononn_engine::helpers::Directory::get_mononn_new_temp_dir();
    mononn_engine::helpers::Directory::create_recursive(debug_dir);
    LOG(ERROR) << "Dump internal state for debugging.";
    for (auto const& [file_name, file_content] : file_collection_list[idx]) {
      std::string full_file_name =
          mononn_engine::helpers::Path::join(debug_dir, file_name);
      mononn_engine::helpers::File::write_to_file(file_content, full_file_name);
    }

    std::string ptx_file_name =
        mononn_engine::helpers::Path::join(debug_dir, "mononn.ptx");
    std::string graph_spec_file_name =
        mononn_engine::helpers::Path::join(debug_dir, "graph_spec.json");
    GraphSpecification* debug_graph_spec = std::get<0>(result_list[idx]);
    mononn_engine::helpers::File::write_to_file(ptx, ptx_file_name);
    mononn_engine::helpers::save_proto_to_json_file(debug_graph_spec,
                                                    graph_spec_file_name);
    LOG(FATAL) << "Kernel create failed for node "
               << debug_graph_spec->codegen_allow_list().at(0)
               << ". Debug directory: " << debug_dir;
  };

  for (int idx = 0; idx < work_count; ++idx) {
    auto compilation_result = finish_queue.wait();

    std::get<0>(result_list[idx]) = reinterpret_cast<GraphSpecification*>(
        compilation_result.tuning_spec_id);
    std::get<1>(result_list[idx]) = std::move(compilation_result.ptx);
    file_collection_list[idx] = std::move(compilation_result.file_collection);
  }

  auto profiling_result = profiling_session.profiling_context([&] {
    for (int idx = 0; idx < work_count; ++idx) {
      const std::string& ptx = std::get<1>(result_list[idx]);
      const auto& cuda_context_proto =
          std::get<0>(result_list[idx])->cuda_context();

      tensorflow::StatusOr<std::unique_ptr<stream_executor::KernelBase>>
          kernel_status_or = xla::gpu::CreateKernel(
              params.kernel_name, params.execution_parameters.size(), ptx, {},
              params.stream->parent());

      std::unique_ptr<stream_executor::KernelBase> kernel =
          std::move(kernel_status_or.ConsumeValueOrDie());

      // Set dynamic shared memory size.
      GpuDriver::FuncSetAttribute(
          CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
          stream_executor::gpu::AsGpuKernel(kernel.get())
              ->AsGpuFunctionHandle(),
          cuda_context_proto.cuda_runtime_context().smem_size());

      std::unique_ptr<stream_executor::KernelArgsArrayBase> kernel_args;

      static constexpr int kKernelArgsLimit = 1024;

      if (params.execution_parameters.size() <= 64) {
        kernel_args = MakeKernelArgs<64>(
            params.execution_parameters,
            cuda_context_proto.cuda_runtime_context().smem_size());
      } else if (params.execution_parameters.size() <= 256) {
        kernel_args = MakeKernelArgs<256>(
            params.execution_parameters,
            cuda_context_proto.cuda_runtime_context().smem_size());
      } else {
        kernel_args = MakeKernelArgs<kKernelArgsLimit>(
            params.execution_parameters,
            cuda_context_proto.cuda_runtime_context().smem_size());
      }

      auto const& block_dim =
          cuda_context_proto.cuda_runtime_context().block_dim();
      auto const& grid_dim =
          cuda_context_proto.cuda_runtime_context().grid_dim();

      tensorflow::Status launch_status = params.stream->parent()->Launch(
          params.stream,
          stream_executor::ThreadDim(block_dim.x(), block_dim.y(),
                                     block_dim.z()),
          stream_executor::BlockDim(grid_dim.x(), grid_dim.y(), grid_dim.z()),
          *kernel, *kernel_args);

      if (!launch_status.ok()) {
        LOG(ERROR) << "Kernel launch failure, "
                   << launch_status.error_message();
        LOG(ERROR) << "# of parameters " << params.execution_parameters.size();
        LOG(ERROR) << "Dynamic smem size in bytes: "
                   << cuda_context_proto.cuda_runtime_context().smem_size();

        error_handle_and_crash(launch_status, ptx, idx);
      }

      tensorflow::Status sync_status = params.stream->BlockHostUntilDone();

      if (!sync_status.ok()) {
        LOG(ERROR) << "Stream synchronize error, "
                   << sync_status.error_message();
        error_handle_and_crash(sync_status, ptx, idx);
      }
    }
  });

  for (int idx = 0; idx < work_count; ++idx) {
    std::get<2>(result_list[idx]) = profiling_result.get_time_in_us(idx);
  }

  return result_list;
}

std::string context_to_str(const CUDAContextProto* cuda_context) {
  auto& grid_dim = cuda_context->cuda_runtime_context().grid_dim();
  auto& block_dim = cuda_context->cuda_runtime_context().block_dim();

  std::string result = mononn_engine::helpers::string_format(
      "%d_%d_%d__%d_%d_%d", grid_dim.x(), grid_dim.y(), grid_dim.z(),
      block_dim.x(), block_dim.y(), block_dim.z());

  return result;
}

// Only consider operator.
std::string get_gemm_or_conv_problem_hash(const Op* op) {
  std::string problem_hash;

  if (op->is_gemm()) {
    problem_hash = "_gemm_";
  } else if (op->is_conv()) {
    problem_hash = "_conv_";
  } else {
    LOG(FATAL) << "Invalid node: " << op->get_name();
  }

  problem_hash +=
      op->get_operand(0)->get_output_spec(0).get_dtype().to_string();
  problem_hash += op->get_operand(0)->get_output_spec(0).to_string();

  problem_hash +=
      op->get_operand(1)->get_output_spec(0).get_dtype().to_string();
  problem_hash += op->get_operand(1)->get_output_spec(0).to_string();

  problem_hash += op->get_output_spec(0).to_string();
  problem_hash += op->get_output_spec(0).to_string();
  return problem_hash;
}

void check_profiling_threadpool(
    CompilationThreadpool& compilation_threadpool,
    ConcurrentQueue<CompilationThreadpool::Result>& finish_queue) {
  if (compilation_threadpool.num_remaining_tasks() != 0) {
    LOG(FATAL) << "Unexpected unfinished tasks";
  }

  if (finish_queue.size() != 0) {
    LOG(FATAL) << "Unexpected unconsumed result in queue";
  }
}

void dump_performance_report(
    std::unordered_map<std::string, std::unordered_map<std::string, double>>&
        best_time_by_context_by_node,
    const std::string& hlo_module_name) {
  auto TF_MONONN_DUMP_DIR =
      mononn_engine::helpers::EnvVar::get_with_default("TF_MONONN_DUMP_DIR");
  std::stringstream file_content;
  if (!TF_MONONN_DUMP_DIR.empty()) {
    for (auto const& [context_str, time_by_node] :
         best_time_by_context_by_node) {
      for (auto const& [node_name, time] : time_by_node) {
        file_content << context_str << "," << node_name << "," << time << "\n";
      }
    }

    std::string cluster_dir = mononn_engine::helpers::Path::join(
        TF_MONONN_DUMP_DIR,
        mononn_engine::helpers::get_hlo_module_short_name(hlo_module_name));
    std::string performance_report_file_name =
        mononn_engine::helpers::Path::join(cluster_dir,
                                           "performance_report.csv");

    std::ofstream ofs(performance_report_file_name);
    ofs << file_content.str();
    ofs.close();
  }
}

std::unique_ptr<MonoNNModule> final_tuning(
    ConcurrentQueue<CompilationThreadpool::Result>& finish_queue,
    CompilationThreadpool& compilation_threadpool,
    const MonoNNModuleTuner::Params& params,
    std::unordered_map<std::string,
                       std::unordered_map<std::string, GraphSpecification*>>&
        optimal_specs_by_context_by_node) {
  auto TF_MONONN_DUMP_DIR =
      mononn_engine::helpers::EnvVar::get_with_default("TF_MONONN_DUMP_DIR");

  std::string cluster_dir_name = mononn_engine::helpers::Path::join(
      TF_MONONN_DUMP_DIR, mononn_engine::helpers::get_hlo_module_short_name(
                              params.hlo_module->name()));
  if (!TF_MONONN_DUMP_DIR.empty()) {
    mononn_engine::helpers::Directory::create_recursive(TF_MONONN_DUMP_DIR);
    mononn_engine::helpers::Directory::create(cluster_dir_name);
  }

  std::unordered_map<std::string, std::unique_ptr<GraphSpecification>>
      optimal_spec_for_each_context;

  for (auto& [context_str, specs_by_node] : optimal_specs_by_context_by_node) {
    std::unique_ptr<GraphSpecification> optimal_spec_for_context =
        mononn_engine::helpers::deep_copy_graph_specification(
            specs_by_node.begin()->second);
    for (auto node_name : optimal_spec_for_context->codegen_reject_list()) {
      optimal_spec_for_context->mutable_codegen_allow_list()->Add(
          std::move(node_name));
    }

    optimal_spec_for_context->mutable_codegen_reject_list()->Clear();

    for (auto& [node_name, spec] : specs_by_node) {
      if (optimal_spec_for_context->gemm_spec_list().contains(node_name)) {
        (*optimal_spec_for_context->mutable_gemm_spec_list())[node_name] =
            spec->gemm_spec_list().at(node_name);
      } else if (optimal_spec_for_context->conv_spec_list().contains(
                     node_name)) {
        (*optimal_spec_for_context->mutable_conv_spec_list())[node_name] =
            spec->conv_spec_list().at(node_name);
      } else if (optimal_spec_for_context->cluster_elewise_spec().contains(
                     node_name)) {
        (*optimal_spec_for_context->mutable_cluster_elewise_spec())[node_name] =
            spec->cluster_elewise_spec().at(node_name);
      } else if (optimal_spec_for_context->cluster_reduce_spec().contains(
                     node_name)) {
        (*optimal_spec_for_context->mutable_cluster_reduce_spec())[node_name] =
            spec->cluster_reduce_spec().at(node_name);
      } else {
        LOG(FATAL) << "Node " << node_name << "not found.";
      }
    }

    std::unique_ptr<GraphSpecification> tuning_spec =
        mononn_engine::helpers::deep_copy_proto<GraphSpecification>(
            optimal_spec_for_context.get());
    TuningSpecId tuning_spec_id =
        reinterpret_cast<TuningSpecId>(optimal_spec_for_context.get());

    if (!TF_MONONN_DUMP_DIR.empty()) {
      std::string cuda_context_str =
          context_to_str(&optimal_spec_for_context->cuda_context());
      std::string json_file_name = mononn_engine::helpers::Path::join(
          cluster_dir_name, cuda_context_str + ".json");
      // std::string proto_file_name = mononn_engine::helpers::Path::join(
      //     cluster_dir_name, cuda_context_str + ".pb");
      mononn_engine::helpers::save_proto_to_json_file(
          optimal_spec_for_context.get(), json_file_name);
      // mononn_engine::helpers::save_proto_to_binary_file(
      //     optimal_spec_for_context.get(), proto_file_name);
    }

    compilation_threadpool.post(
        params.hlo_module, tuning_spec_id, std::move(tuning_spec),
        params.kernel_name, params.allocation_list,
        &params.buffer_assignment->alias_analysis(), finish_queue,
        CompilationThreadpool::COMPILATION_SAVE_SOURCE);

    optimal_spec_for_each_context[context_str] =
        std::move(optimal_spec_for_context);
  }

  std::stringstream ss_perf_result;

  ss_perf_result << "Tuning result for each thread grid setting:\n";

  double best_time_in_us = -1;
  GraphSpecification* best_tuning_spec = nullptr;
  std::string best_tuning_spec_ptx;

  for (int idx = 0; idx < optimal_spec_for_each_context.size(); ++idx) {
    auto compilation_result = finish_queue.wait();
    GraphSpecification* tuning_spec = reinterpret_cast<GraphSpecification*>(
        compilation_result.tuning_spec_id);

    CuptiProfilingSession::ProfilingResult profiling_result = profile_kernel(
        params, tuning_spec->cuda_context(), compilation_result.ptx, {});

    std::string cuda_context_str = context_to_str(&tuning_spec->cuda_context());
    ss_perf_result << cuda_context_str << " "
                   << profiling_result.get_time_in_ms(0) << " ms\n";

    if (best_tuning_spec == nullptr ||
        (profiling_result.get_time_in_us(0) < best_time_in_us)) {
      best_time_in_us = profiling_result.get_time_in_us(0);
      best_tuning_spec = tuning_spec;
      best_tuning_spec_ptx = compilation_result.ptx;
    }

    // Save ptx and source code
    if (!TF_MONONN_DUMP_DIR.empty()) {
      std::string ptx_file_name = mononn_engine::helpers::Path::join(
          cluster_dir_name, cuda_context_str + ".ptx");
      mononn_engine::helpers::File::write_to_file(compilation_result.ptx,
                                                  ptx_file_name);

      // Save source
      if (!compilation_result.file_collection.empty()) {
        std::string source_dir_name = mononn_engine::helpers::Path::join(
            cluster_dir_name, cuda_context_str + "_src");
        mononn_engine::helpers::Directory::create(source_dir_name);
        for (auto const& [file_name, file_content] :
             compilation_result.file_collection) {
          std::string full_file_name =
              mononn_engine::helpers::Path::join(source_dir_name, file_name);
          mononn_engine::helpers::File::write_to_file(file_content,
                                                      full_file_name);
        }
      }
    }
  }

  LOG(INFO) << ss_perf_result.str();

  if (!TF_MONONN_DUMP_DIR.empty()) {
    std::string json_file_name = mononn_engine::helpers::Path::join(
        cluster_dir_name, "best_tuning_spec.json");
    std::string proto_file_name = mononn_engine::helpers::Path::join(
        cluster_dir_name, "best_tuning_spec.pb");
    std::string ptx_file_name = mononn_engine::helpers::Path::join(
        cluster_dir_name, "best_tuning_spec.ptx");

    mononn_engine::helpers::save_proto_to_json_file(best_tuning_spec,
                                                    json_file_name);
    mononn_engine::helpers::save_proto_to_binary_file(best_tuning_spec,
                                                      proto_file_name);
    mononn_engine::helpers::File::write_to_file(best_tuning_spec_ptx,
                                                ptx_file_name);

    std::string perf_result_dump_file =
        mononn_engine::helpers::Path::join(cluster_dir_name, "tuning_log.log");
    mononn_engine::helpers::File::write_to_file(ss_perf_result.str(),
                                                perf_result_dump_file);
  }

  check_profiling_threadpool(compilation_threadpool, finish_queue);

  auto best_tuning_spec_unique =
      mononn_engine::helpers::deep_copy_proto<GraphSpecification>(
          best_tuning_spec);

  std::unique_ptr<MonoNNModule> mononn_module = std::make_unique<MonoNNModule>(
      params.hlo_module, std::move(best_tuning_spec_unique), params.kernel_name,
      params.allocation_list, &params.buffer_assignment->alias_analysis());

  mononn_module->set_ptx(best_tuning_spec_ptx);

  return std::move(mononn_module);
}

std::unique_ptr<MonoNNModule> MonoNNModuleTuner::tune(const Params& params) {
  LOG(INFO) << "====================MonoNN tuning, cluster name "
            << params.hlo_module->name() << "====================";

  std::unique_ptr<Graph> graph =
      IRParserFused::from_hlo_module_unique(params.hlo_module);
  CompilationThreadpool compilation_threadpool;
  ConcurrentQueue<CompilationThreadpool::Result> finish_queue;
  std::unordered_map<std::string, std::unordered_map<std::string, double>>
      best_time_by_context_by_node;
  std::unordered_map<std::string,
                     std::unordered_map<std::string, GraphSpecification*>>
      graph_spec_by_context_by_node;

  OptimizationRunner::run_group_pre_impl_assignment(graph.get(), nullptr);

  std::vector<std::unique_ptr<GraphSpecification>> candidate_tuning_spec_list =
      std::move(TuningSpaceGenerator::generate_tuning_space(graph.get(), "", {},
                                                            {}, {}));

  constexpr int profiling_batch_size = 100;

  {
    int non_gemm_nodes_task_count = 0;

    LOG(INFO) << "Begin profiling non GEMM/Conv nodes";
    LOG(INFO) << "Enqueue tasks.";
    for (auto const& graph_spec : candidate_tuning_spec_list) {
      if (graph_spec->codegen_allow_list_size() != 1) {
        LOG(FATAL) << "Codegen allow list is not one: ";
      }

      const std::string& node_name = graph_spec->codegen_allow_list(0);

      if (graph->get_node(node_name)->is_gemm() ||
          graph->get_node(node_name)->is_conv()) {
        continue;
      } else {
        non_gemm_nodes_task_count++;
      }

      std::unique_ptr<GraphSpecification> tuning_spec =
          mononn_engine::helpers::deep_copy_proto<GraphSpecification>(
              graph_spec.get());

      TuningSpecId tuning_spec_id =
          reinterpret_cast<TuningSpecId>(graph_spec.get());
      compilation_threadpool.post(
          params.hlo_module, tuning_spec_id, std::move(tuning_spec),
          params.kernel_name, params.allocation_list,
          &params.buffer_assignment->alias_analysis(), finish_queue,
          CompilationThreadpool::COMPILATION_SAVE_SOURCE);
    }

    LOG(INFO) << "Enqueue complete. " << non_gemm_nodes_task_count
              << " in total.";

    for (int idx = 0; idx < non_gemm_nodes_task_count;
         idx += profiling_batch_size) {
      int work_count =
          std::min(profiling_batch_size, non_gemm_nodes_task_count - idx);

      std::vector<std::tuple<GraphSpecification*, std::string, double>>
          batched_profiling_result =
              profile_in_batch(finish_queue, work_count, params);

      // std::pair<TuningSpecId, std::string> ptx_result = finish_queue.wait();
      // GraphSpecification *tuning_spec = reinterpret_cast<GraphSpecification
      // *>(ptx_result.first);

      // std::string context_str = context_to_str(&tuning_spec->cuda_context());
      // const std::string &node_name = tuning_spec->codegen_allow_list(0);

      // CuptiProfilingSession::ProfilingResult profiling_result
      //         = profile_kernel(params, tuning_spec->cuda_context(),
      //         ptx_result.second, {});

      // if (idx % 100 == 0 || idx == non_gemm_nodes_task_count - 1) {
      //     LOG(INFO) << "Non GEMM/Conv node profiling progres " << idx + 1 <<
      //     "/" << non_gemm_nodes_task_count;
      // }

      LOG(INFO) << "Memory intensive nodes profiling progres "
                << idx + work_count << "/" << non_gemm_nodes_task_count;

      for (auto const& result_item : batched_profiling_result) {
        GraphSpecification* tuning_spec = std::get<0>(result_item);
        std::string context_str = context_to_str(&tuning_spec->cuda_context());
        const std::string& node_name = tuning_spec->codegen_allow_list(0);
        double time_in_us = std::get<2>(result_item);
        if (!best_time_by_context_by_node.count(context_str)) {
          best_time_by_context_by_node[context_str] =
              std::unordered_map<std::string, double>();
          graph_spec_by_context_by_node[context_str] =
              std::unordered_map<std::string, GraphSpecification*>();
        }

        if ((!best_time_by_context_by_node[context_str].count(node_name)) ||
            best_time_by_context_by_node[context_str][node_name] > time_in_us) {
          best_time_by_context_by_node[context_str][node_name] = time_in_us;
          graph_spec_by_context_by_node[context_str][node_name] = tuning_spec;
        }
      }
    }
  }

  check_profiling_threadpool(compilation_threadpool, finish_queue);

  LOG(INFO) << "End profiling non GEMM/Conv nodes";
  LOG(INFO) << "Begin profiling GEMM/Conv nodes";

  {
    std::unordered_map<std::string, std::vector<GraphSpecification*>>
        gemm_conv_spec_by_node;
    std::unordered_set<std::string> node_hash_done;
    int gemm_conv_profiling_task_count = 0;

    LOG(INFO) << "Enqueue tasks.";

    for (const auto& graph_spec : candidate_tuning_spec_list) {
      const std::string& node_name = graph_spec->codegen_allow_list(0);
      auto node = graph->get_node(node_name);
      if (!(node->is_gemm() || node->is_conv())) {
        continue;
      }

      if (!gemm_conv_spec_by_node.count(node_name)) {
        gemm_conv_spec_by_node[node_name] = std::vector<GraphSpecification*>();
      }

      gemm_conv_spec_by_node[node_name].push_back(graph_spec.get());
    }

    for (auto const& [node_name, graph_spec_list] : gemm_conv_spec_by_node) {
      auto node = graph->get_node(node_name);
      // Only consider node hash here.
      std::string node_problem_hash = get_gemm_or_conv_problem_hash(node.get());

      if (node_hash_done.count(node_problem_hash)) {
        continue;
      }

      node_hash_done.insert(node_problem_hash);

      for (auto const& graph_spec : graph_spec_list) {
        ++gemm_conv_profiling_task_count;
        TuningSpecId tuning_spec_id =
            reinterpret_cast<TuningSpecId>(graph_spec);
        std::unique_ptr<GraphSpecification> tuning_spec =
            mononn_engine::helpers::deep_copy_proto<GraphSpecification>(
                graph_spec);
        compilation_threadpool.post(
            params.hlo_module, tuning_spec_id, std::move(tuning_spec),
            params.kernel_name, params.allocation_list,
            &params.buffer_assignment->alias_analysis(), finish_queue,
            CompilationThreadpool::COMPILATION_SAVE_SOURCE);
      }
    }

    LOG(INFO) << "Enqueue complete. " << gemm_conv_profiling_task_count
              << " in total.";

    std::unordered_map<std::string,
                       std::unordered_map<std::string, CutlassConfigProto>>
        best_cutlass_config_per_context_per_problem_hash;
    std::unordered_map<std::string, std::unordered_map<std::string, double>>
        best_time_per_context_per_problem_hash;

    for (int idx = 0; idx < gemm_conv_profiling_task_count;
         idx += profiling_batch_size) {
      int work_count =
          std::min(profiling_batch_size, gemm_conv_profiling_task_count - idx);

      std::vector<std::tuple<GraphSpecification*, std::string, double>>
          batched_profiling_result =
              profile_in_batch(finish_queue, work_count, params);

      // std::pair<TuningSpecId, std::string> ptx_result = finish_queue.wait();
      // GraphSpecification *tuning_spec = reinterpret_cast<GraphSpecification
      // *>(ptx_result.first);

      // CuptiProfilingSession::ProfilingResult profiling_result
      //         = profile_kernel( params, tuning_spec->cuda_context(),
      //         ptx_result.second, {});

      // if (idx % 100 == 0 || idx == gemm_conv_profiling_task_count - 1) {
      //     LOG(INFO) << "GEMM/Conv node profiling progres " << idx + 1 << "/"
      //     << gemm_conv_profiling_task_count;
      // }

      // const std::string &node_name = tuning_spec->codegen_allow_list(0);
      // auto node = graph->get_node(node_name);
      // std::string context_str = context_to_str(&tuning_spec->cuda_context());
      // std::string problem_hash = get_gemm_or_conv_problem_hash(node.get());

      // time_by_problem_hash[problem_hash] = profiling_result.get_time_in_us();

      LOG(INFO) << "Compute intensive nodes profiling progres "
                << idx + work_count << "/" << gemm_conv_profiling_task_count;

      for (auto const& result_item : batched_profiling_result) {
        GraphSpecification* tuning_spec = std::get<0>(result_item);
        std::string context_str = context_to_str(&tuning_spec->cuda_context());
        const std::string& node_name = tuning_spec->codegen_allow_list(0);
        auto node = graph->get_node(node_name);
        std::string problem_hash = get_gemm_or_conv_problem_hash(node.get());
        double time_in_us = std::get<2>(result_item);

        if (!best_time_per_context_per_problem_hash.count(context_str)) {
          best_time_per_context_per_problem_hash[context_str] =
              std::unordered_map<std::string, double>();
          best_cutlass_config_per_context_per_problem_hash[context_str] =
              std::unordered_map<std::string, CutlassConfigProto>();
        }

        if ((!best_time_per_context_per_problem_hash[context_str].count(
                problem_hash)) ||
            best_time_per_context_per_problem_hash[context_str][problem_hash] >
                time_in_us) {
          // if
          // (best_cutlass_config_per_context_per_problem_hash[context_str].count(problem_hash))
          // {
          //     auto old_inst_shape =
          //     best_spec_per_context_per_problem_hash[context_str][problem_hash]->gemm_spec_list().at(node_name).cutlass_config().instructionshape();
          //     auto inst_shape =
          //     tuning_spec->gemm_spec_list().at(node_name).cutlass_config().instructionshape();

          //     if (inst_shape.m() == 1 && inst_shape.n() == 1 &&
          //     inst_shape.k() == 1 && old_inst_shape.m() != 1) LOG(DEBUG) <<
          //     "Drop tensorop instruction: " << node_name <<
          //     old_inst_shape.m() << " " << old_inst_shape.n() << " " <<
          //     old_inst_shape.k() << " Time: "
          //         <<
          //         best_time_per_context_per_problem_hash[context_str][problem_hash]
          //         << " " << time_in_us;
          // }

          best_time_per_context_per_problem_hash[context_str][problem_hash] =
              time_in_us;
          best_cutlass_config_per_context_per_problem_hash
              [context_str][problem_hash] =
                  extract_cutlass_config_proto(tuning_spec);
        }
        // else {
        //     auto old_inst_shape =
        //     best_spec_per_context_per_problem_hash[context_str][problem_hash]->gemm_spec_list().at(node_name).cutlass_config().instructionshape();
        //     auto inst_shape =
        //     tuning_spec->gemm_spec_list().at(node_name).cutlass_config().instructionshape();
        //     if (old_inst_shape.m() == 1 && old_inst_shape.n() == 1 &&
        //     old_inst_shape.k() == 1 && inst_shape.m() != 1) {
        //         LOG(DEBUG) << "Tensor op operation cannot beat simt
        //         operation: context " << context_str << " " << node_name << "
        //         time " <<
        //         best_time_per_context_per_problem_hash[context_str][problem_hash]
        //         << " " << time_in_us
        //             << " " << inst_shape.m() << " " << inst_shape.n() << " "
        //             << inst_shape.k();
        //     }
        // }
      }
    }

    // for (auto const &[context_str, problem_hash_to_spec] :
    // best_spec_per_context_per_problem_hash) {
    //     LOG(DEBUG) << "======Context str=======" << context_str;
    //     for (auto const &[problem_hash, spec] : problem_hash_to_spec) {
    //         const std::string &node_name = spec->codegen_allow_list(0);
    //         auto inst_shape =
    //         spec->gemm_spec_list().at(node_name).cutlass_config().instructionshape();
    //         LOG(DEBUG) << "\t" << problem_hash << " " << inst_shape.m() << "
    //         " << inst_shape.n() << " " << inst_shape.k();
    //     }
    // }

    check_profiling_threadpool(compilation_threadpool, finish_queue);

    for (const auto& graph_spec : candidate_tuning_spec_list) {
      std::string context_str = context_to_str(&graph_spec->cuda_context());
      const std::string& node_name = graph_spec->codegen_allow_list(0);
      auto node = graph->get_node(node_name);
      if (!(node->is_gemm() || node->is_conv())) {
        continue;
      }

      std::string problem_hash = get_gemm_or_conv_problem_hash(node.get());
      // double time_in_us = time_by_problem_hash[problem_hash];

      if (!best_time_per_context_per_problem_hash.count(context_str)) {
        LOG(FATAL) << "best_time_per_context_per_problem_hash do not have "
                   << context_str;
      }

      if (!best_time_per_context_per_problem_hash[context_str].count(
              problem_hash)) {
        LOG(FATAL) << "best_time_per_context_per_problem_hash[" << context_str
                   << "] do not have " << problem_hash;
      }

      // Already set
      if (best_time_by_context_by_node.count(context_str) &&
          best_time_by_context_by_node[context_str].count(node_name)) {
        continue;
      }

      if (!best_time_by_context_by_node.count(context_str)) {
        best_time_by_context_by_node[context_str] =
            std::unordered_map<std::string, double>();
        graph_spec_by_context_by_node[context_str] =
            std::unordered_map<std::string, GraphSpecification*>();
      }

      auto best_spec = std::find_if(
          gemm_conv_spec_by_node[node_name].begin(),
          gemm_conv_spec_by_node[node_name].end(),
          [&](const GraphSpecification* tmp_spec) -> bool {
            using MessageDifferencer =
                google::protobuf::util::MessageDifferencer;

            bool cuda_context_match = MessageDifferencer::Equals(
                graph_spec->cuda_context(), tmp_spec->cuda_context());
            bool is_gemm_and_cutlass_config_match =
                node->is_gemm() &&
                MessageDifferencer::Equals(
                    best_cutlass_config_per_context_per_problem_hash
                        [context_str][problem_hash],
                    tmp_spec->gemm_spec_list().at(node_name).cutlass_config());
            bool is_conv_and_cutlass_config_match =
                node->is_conv() &&
                MessageDifferencer::Equals(
                    best_cutlass_config_per_context_per_problem_hash
                        [context_str][problem_hash],
                    tmp_spec->conv_spec_list().at(node_name).cutlass_config());
            return (cuda_context_match && (is_gemm_and_cutlass_config_match ||
                                           is_conv_and_cutlass_config_match));
          });

      if (best_spec == gemm_conv_spec_by_node[node_name].end()) {
        LOG(FATAL) << context_str << " " << node_name << " not found.";
      }

      // auto reference_inst_shape =
      // best_spec_per_context_per_problem_hash[context_str][problem_hash]->gemm_spec_list().at(node_name).cutlass_config().instructionshape();

      // auto _inst_shape =
      // (*best_spec)->gemm_spec_list().at(node_name).cutlass_config().instructionshape();

      // LOG(DEBUG) << "Node:" << node_name << " context " << context_str << "
      // Best spec: " << _inst_shape.m() << " " << _inst_shape.n() << " " <<
      // _inst_shape.k()
      //     << " Reference instruction shape: " << reference_inst_shape.m() <<
      //     " " << reference_inst_shape.n() << " " << reference_inst_shape.k();

      best_time_by_context_by_node[context_str][node_name] =
          best_time_per_context_per_problem_hash[context_str][problem_hash];
      graph_spec_by_context_by_node[context_str][node_name] = *best_spec;
    }
  }

  LOG(INFO) << "Begin final tuning...";

  std::unique_ptr<MonoNNModule> mononn_module =
      final_tuning(finish_queue, compilation_threadpool, params,
                   graph_spec_by_context_by_node);

  check_profiling_threadpool(compilation_threadpool, finish_queue);

  LOG(INFO) << "Tuning complete.";

  dump_performance_report(best_time_by_context_by_node,
                          params.hlo_module->name());
  return std::move(mononn_module);
}
}  // namespace module
}  // namespace mononn_engine