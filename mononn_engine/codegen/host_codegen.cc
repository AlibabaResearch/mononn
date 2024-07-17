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

#include "mononn_engine/codegen/host_codegen.h"

#include <sstream>
#include <vector>

#include "mononn_engine/config/config.h"
#include "mononn_engine/core/gpu/buffer_manager.h"
#include "mononn_engine/core/gpu/multi_buffer.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/op/parameter.h"
#include "mononn_engine/core/semantic/cuda_invocation.h"
#include "mononn_engine/core/semantic/function_invocation.h"
#include "mononn_engine/core/tensor/tensor_spec.h"
#include "mononn_engine/helpers/path.h"
#include "mononn_engine/helpers/string_helpers.h"
#include "tensorflow/core/platform/env.h"

namespace mononn_engine {
namespace codegen {
using FunctionInvocation = mononn_engine::core::semantic::FunctionInvocation;
using BufferManager = mononn_engine::core::gpu::BufferManager;
using CUDAInvocation = mononn_engine::core::semantic::CUDAInvocation;
using OpType = mononn_engine::core::op::OpType;
using Config = mononn_engine::config::Config;
using Path = mononn_engine::helpers::Path;
using Parameter = mononn_engine::core::op::Parameter;
using TensorSpec = mononn_engine::core::tensor::TensorSpec;
using MultiBuffer = mononn_engine::core::gpu::MultiBuffer;

std::string HostCodegen::generate(std::shared_ptr<CUDAContext> cuda_context,
                                  Graph* graph,
                                  const std::string& kernel_name) {
  std::stringstream ss;
  ss << "int main(int argc, char const *argv[]) {"
     << "\n";
  ss << HostCodegen::generate_stream(cuda_context, graph);
  ss << HostCodegen::generate_memory_allocation(cuda_context, graph);

  if (std::find(Config::get()->host_codegen_disabled_pass.begin(),
                Config::get()->host_codegen_disabled_pass.end(),
                "generate_memory_initialization") ==
      Config::get()->host_codegen_disabled_pass.end()) {
    ss << HostCodegen::generate_memory_initialization(cuda_context, graph);
  }

  if (std::count(Config::get()->host_codegen_disabled_pass.begin(),
                 Config::get()->host_codegen_disabled_pass.end(),
                 "generate_parameter_initialization") == 0) {
    ss << HostCodegen::generate_parameter_declaration(cuda_context, graph);
  }

  ss << HostCodegen::generate_kernel_invocation(cuda_context, graph,
                                                kernel_name);
  ss << HostCodegen::generate_stream_synchronize(
      cuda_context->cuda_runtime_context.stream);
  //        ss << HostCodegen::generate_print_output(cuda_context, graph);
  ss << "return 0;"
     << "\n";
  ss << "}"
     << "\n";
  return ss.str();
}

std::string HostCodegen::generate_memory_initialization(
    std::shared_ptr<CUDAContext> cuda_context, Graph* graph) {
  std::stringstream ss;

  std::vector<std::string> buffer_list;

  for (auto const& node_name : BufferManager::get_buffered_nodes_in_global()) {
    if (!graph->has_node(node_name)) continue;
    buffer_list.push_back(node_name);
  }

  for (auto const& node_name : buffer_list) {
    std::shared_ptr<Op> node = graph->get_node(node_name);

    if (node->get_type() == OpType::constant) {
      auto dtype = node->get_output_spec(0).get_dtype();

      ss << mononn_engine::helpers::string_format(
                "auto %s = cnpy::npy_load(\"%s\");", node_name.c_str(),
                ("./" + node_name + ".npy").c_str())
         << "\n";
      FunctionInvocation check_error("checkCudaErrors");
      FunctionInvocation cuda_memcpy("cudaMemcpyAsync");

      cuda_memcpy.add_arg(BufferManager::get_buffer_name(node_name));
      cuda_memcpy.add_arg(mononn_engine::helpers::string_format(
          "%s.data<%s>()", node_name.c_str(),
          dtype.to_cutlass_type().to_string().c_str()));
      cuda_memcpy.add_arg(
          std::to_string(node->get_output_buffer_size_in_bytes()));
      cuda_memcpy.add_arg("cudaMemcpyHostToDevice");
      cuda_memcpy.add_arg(cuda_context->cuda_runtime_context.stream);

      check_error.add_arg(cuda_memcpy.to_string());

      ss << check_error.to_string() << ";\n";
    } else {
      FunctionInvocation check_error("checkCudaErrors");
      FunctionInvocation cuda_memset("cudaMemsetAsync");

      cuda_memset.add_arg(BufferManager::get_buffer_name(node_name));
      cuda_memset.add_arg("0");
      cuda_memset.add_arg(
          std::to_string(node->get_output_buffer_size_in_bytes()));
      cuda_memset.add_arg(cuda_context->cuda_runtime_context.stream);
      check_error.add_arg(cuda_memset.to_string());

      ss << check_error.to_string() << ";\n";
    }
  }

  return ss.str();
}

std::string HostCodegen::generate_parameter_declaration(
    std::shared_ptr<CUDAContext> cuda_context, Graph* graph) {
  if (Config::get()->feeds.empty()) {
    LOG(FATAL) << "Input nodes unspecified";
  }

  std::vector<std::string> feeds = Config::get()->feeds;
  std::vector<std::string> sorted_feeds = feeds;
  std::sort(sorted_feeds.begin(), sorted_feeds.end());

  std::stringstream ss;
  std::vector<std::string> buffer_list;

  for (auto const& node_name : BufferManager::get_buffered_nodes_in_global()) {
    if (!graph->has_node(node_name)) continue;
    buffer_list.push_back(node_name);
  }

  for (auto const& node_name : buffer_list) {
    std::shared_ptr<Op> node = graph->get_node(node_name);

    if (node->get_type() == OpType::parameter) {
      int parameter_id = node->as<Parameter>()->get_parameter_number();
      std::string feed = sorted_feeds[parameter_id];
      int feed_id = std::find(feeds.begin(), feeds.end(), feed) - feeds.begin();
      std::string input_data_file = Config::get()->input_data_files[feed_id];

      auto dtype = node->get_output_spec(0).get_dtype();
      ss << "// load argument for " << feed << std::endl;
      ss << mononn_engine::helpers::string_format(
                "auto %s = cnpy::npy_load(\"%s\");", node_name.c_str(),
                input_data_file.c_str())
         << "\n";
    }
  }

  return ss.str();
}

std::string HostCodegen::generate_parameter_initialization(
    std::shared_ptr<CUDAContext> cuda_context, Graph* graph) {
  std::stringstream ss;

  std::vector<std::string> buffer_list;

  for (auto const& node_name : BufferManager::get_buffered_nodes_in_global()) {
    if (!graph->has_node(node_name)) continue;
    buffer_list.push_back(node_name);
  }

  for (auto const& node_name : buffer_list) {
    std::shared_ptr<Op> node = graph->get_node(node_name);

    if (node->get_type() == OpType::parameter) {
      auto dtype = node->get_output_spec(0).get_dtype();
      FunctionInvocation check_error("checkCudaErrors");
      FunctionInvocation cuda_memcpy("cudaMemcpyAsync");

      cuda_memcpy.add_arg(BufferManager::get_buffer_name(node_name));
      cuda_memcpy.add_arg(mononn_engine::helpers::string_format(
          "%s.data<%s>()", node_name.c_str(),
          dtype.to_cutlass_type().to_string().c_str()));
      cuda_memcpy.add_arg(
          std::to_string(node->get_output_buffer_size_in_bytes()));
      cuda_memcpy.add_arg("cudaMemcpyHostToDevice");
      cuda_memcpy.add_arg(cuda_context->cuda_runtime_context.stream);

      check_error.add_arg(cuda_memcpy.to_string());

      ss << check_error.to_string() << ";\n";
    }
  }

  return ss.str();
}

std::string HostCodegen::generate_stream(
    std::shared_ptr<CUDAContext> cuda_context, Graph* graph) {
  std::stringstream ss;
  ss << mononn_engine::helpers::string_format(
            "cudaStream_t %s;",
            cuda_context->cuda_runtime_context.stream.c_str())
     << "\n";
  ss << mononn_engine::helpers::string_format(
            "checkCudaErrors(cudaStreamCreateWithFlags(&%s, "
            "cudaStreamNonBlocking));",
            cuda_context->cuda_runtime_context.stream.c_str())
     << "\n";

  return ss.str();
}

std::string HostCodegen::generate_memory_allocation(
    std::shared_ptr<CUDAContext> cuda_context, Graph* graph) {
  std::vector<std::string> buffer_node_list;
  std::stringstream ss;

  for (auto const& node_name : BufferManager::get_buffered_nodes_in_global()) {
    if (!graph->has_node(node_name)) continue;
    buffer_node_list.push_back(node_name);
  }

  std::string onefuser_memory_buffer_name = Config::get()->onefuser_buffer_name;
  MultiBuffer onefuser_memory_buffer(onefuser_memory_buffer_name);

  for (auto const& node_name : buffer_node_list) {
    std::shared_ptr<Op> node = graph->get_node(node_name);
    onefuser_memory_buffer.add_buffer(node->get_output_buffer_size_in_bytes());
  }

  FunctionInvocation check_error("checkCudaErrors");
  FunctionInvocation cuda_malloc("cudaMallocAsync");

  cuda_malloc.add_arg("&" + onefuser_memory_buffer_name);
  cuda_malloc.add_arg(
      std::to_string(onefuser_memory_buffer.get_total_size_in_bytes()));
  cuda_malloc.add_arg(cuda_context->cuda_runtime_context.stream);

  check_error.add_arg(cuda_malloc.to_string());

  ss << "void *" << onefuser_memory_buffer_name << ";\n";
  ss << check_error.to_string() << ";\n";

  for (int idx = 0; idx < (int)buffer_node_list.size(); ++idx) {
    std::string node_name = buffer_node_list[idx];
    std::string buffer_name = BufferManager::get_buffer_name(node_name);

    ss << mononn_engine::helpers::string_format(
              "void *%s = %s;", buffer_name.c_str(),
              onefuser_memory_buffer.get_pointer_to_buffer(idx).c_str())
       << "\n";
  }

  return ss.str();
}

std::string HostCodegen::generate_kernel_invocation(
    std::shared_ptr<CUDAContext> cuda_context, Graph* graph,
    const std::string& kernel_name) {
  CUDAInvocation cuda_invocation(kernel_name,
                                 cuda_context->cuda_runtime_context.grid_dim,
                                 cuda_context->cuda_runtime_context.block_dim,
                                 cuda_context->cuda_runtime_context.smem_size,
                                 cuda_context->cuda_runtime_context.stream);

  cuda_invocation.add_arg(Config::get()->onefuser_buffer_name);

  std::stringstream ss;

  if (cuda_context->cuda_runtime_context.smem_size >= (48 << 10)) {
    FunctionInvocation check_cuda_error("checkCudaErrors");
    FunctionInvocation cuda_func_set_attr("cudaFuncSetAttribute");
    cuda_func_set_attr.add_arg(kernel_name);
    cuda_func_set_attr.add_arg("cudaFuncAttributeMaxDynamicSharedMemorySize");
    cuda_func_set_attr.add_arg(
        std::to_string(cuda_context->cuda_runtime_context.smem_size));

    check_cuda_error.add_arg(cuda_func_set_attr.to_string());

    ss << check_cuda_error.to_string() << ";\n";
  }

  ss << "cudaEvent_t start, stop;"
     << "\n";
  ss << "checkCudaErrors(cudaEventCreate(&start));"
     << "\n";
  ss << "checkCudaErrors(cudaEventCreate(&stop));"
     << "\n";
  ss << mononn_engine::helpers::string_format(
            "checkCudaErrors(cudaEventRecord(start, %s));",
            cuda_context->cuda_runtime_context.stream.c_str())
     << "\n";
  ss << "for (int i = 0; i < 100; ++i) {"
     << "\n";

  if (std::count(Config::get()->host_codegen_disabled_pass.begin(),
                 Config::get()->host_codegen_disabled_pass.end(),
                 "generate_parameter_initialization") == 0) {
    ss << HostCodegen::generate_parameter_initialization(cuda_context, graph);
  }

  ss << cuda_invocation.to_string() << ";\n";
  ss << "}"
     << "\n";
  ss << mononn_engine::helpers::string_format(
            "checkCudaErrors(cudaEventRecord(stop, %s));",
            cuda_context->cuda_runtime_context.stream.c_str())
     << "\n";
  ss << "checkCudaErrors(cudaEventSynchronize(stop));"
     << "\n";

  ss <<
      R"(float milliseconds = 0;
checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
printf("Time in us: %.3f\n", milliseconds * 1000 / 100);
)";

  return ss.str();
}

std::string HostCodegen::generate_print_output(
    std::shared_ptr<CUDAContext> cuda_context, Graph* graph) {
  std::stringstream ss;

  int num_output_nodes = graph->get_output_nodes().size();

  std::function<void(std::string, std::string, std::string, TensorSpec)>
      print_single_output = [&](std::string name, std::string dev_ptr,
                                std::string host_ptr,
                                TensorSpec tensor_spec) -> void {
    auto type = tensor_spec.get_dtype();

    ss << mononn_engine::helpers::string_format(
              "std::array<%s, %s> %s;",
              type.to_cutlass_type().to_string().c_str(),
              std::to_string(tensor_spec.element_count()).c_str(),
              host_ptr.c_str())
       << "\n";

    FunctionInvocation cuda_check_error("checkCudaErrors");
    FunctionInvocation cuda_memcpy("cudaMemcpyAsync");
    cuda_memcpy.add_arg(host_ptr + ".data()");
    cuda_memcpy.add_arg(dev_ptr);
    cuda_memcpy.add_arg(std::to_string(tensor_spec.size_in_bytes()));
    cuda_memcpy.add_arg("cudaMemcpyDeviceToHost");
    cuda_memcpy.add_arg(cuda_context->cuda_runtime_context.stream);

    cuda_check_error.add_arg(cuda_memcpy.to_string());

    ss << cuda_check_error.to_string() << ";\n";
    ss << HostCodegen::generate_stream_synchronize(
        cuda_context->cuda_runtime_context.stream);

    ss << mononn_engine::helpers::string_format("std::cout << \"==========\";")
       << "\n";
    ss << mononn_engine::helpers::string_format(
        "std::cout << \"%s\" << \" \" << \"%s\" << \"%s\";\n", name.c_str(),
        type.to_string().c_str(), tensor_spec.to_string().c_str());
    ss << mononn_engine::helpers::string_format(
              "std::cout << \"==========\" << std::endl;")
       << "\n";

    int row_count = tensor_spec.element_count() / tensor_spec.get_shape(-1);
    int col_count = tensor_spec.get_shape(-1);
    std::string offset =
        mononn_engine::helpers::string_format("c + r * %d", col_count);

    ss << mononn_engine::helpers::string_format(
              "for (int r = 0; r < %d; ++r) {", row_count)
       << "\n";
    ss << "std::cout << \"{ \";"
       << "\n";
    ss << mononn_engine::helpers::string_format(
              "for (int c = 0; c < %d; ++c) {", col_count)
       << "\n";
    ss << "if (c != 0) { std::cout << \", \"; }"
       << "\n";
    ss << mononn_engine::helpers::string_format(
              "std::cout << %s[%s];", host_ptr.c_str(), offset.c_str())
       << "\n";
    ss << "}"
       << "\n";
    ss << "std::cout << \" }\" << std::endl;"
       << "\n";
    ss << "}"
       << "\n";
  };

  for (auto const& output_node_name : graph->get_output_nodes()) {
    std::shared_ptr<Op> output_node = graph->get_node(output_node_name);

    if (output_node->get_output_specs().size() >
        1) {  // cluster multiple output
      MultiBuffer multi_buffer(
          BufferManager::get_buffer_name(output_node_name));

      int output_count = output_node->get_output_specs().size();

      for (int idx = 0; idx < output_count; ++idx) {
        multi_buffer.add_buffer(output_node->get_output_spec(idx));
      }

      for (int idx = 0; idx < output_count; ++idx) {
        std::string name = output_node_name + "_output" + std::to_string(idx);
        std::string dev_ptr = multi_buffer.get_pointer_to_buffer(idx);
        std::string host_buffer =
            output_node_name + "_output" + std::to_string(idx) + "_host";

        print_single_output(name, dev_ptr, host_buffer,
                            output_node->get_output_spec(idx));
      }
    } else {  // cluster single output
      std::string name = output_node_name;
      std::string dev_ptr = BufferManager::get_buffer_name(output_node_name);
      std::string host_buffer = output_node_name + "_host";

      print_single_output(name, dev_ptr, host_buffer,
                          output_node->get_output_spec(0));
    }
  }

  return ss.str();
}

std::string HostCodegen::generate_stream_synchronize(std::string stream) {
  return mononn_engine::helpers::string_format(
      "checkCudaErrors(cudaStreamSynchronize(%s));\n", stream.c_str());
}
}  // namespace codegen
}  // namespace mononn_engine