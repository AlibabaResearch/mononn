#include "mononn_engine/codegen/clustered_graph_codegen.h"

#include <sstream>

#include "mononn_engine/codegen/cuda_program.h"
#include "mononn_engine/core/edge/edge.h"
#include "mononn_engine/core/gpu/buffer_manager.h"
#include "mononn_engine/core/gpu/synchronization.h"
#include "mononn_engine/core/op/cluster_op.h"
#include "mononn_engine/helpers/string_helpers.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace codegen {
using CUDAProgram = mononn_engine::codegen::CUDAProgram;
using BufferManager = mononn_engine::core::gpu::BufferManager;
using ClusterOp = mononn_engine::core::op::ClusterOp;
using Edge = mononn_engine::core::edge::Edge<ClusterOp>;
using Synchronization = mononn_engine::core::gpu::Synchronization;

CUDAProgram ClusteredGraphCodegen::generate(
    std::shared_ptr<CUDAContext> cuda_context,
    std::shared_ptr<ClusteredGraph> graph) {
  CUDAProgram cuda_program(cuda_context);
  ClusteredGraphCodegen::initialize_buffer_manager(graph);
  int block_size = cuda_context->cuda_runtime_context.block_dim.XYZ();
  int block_count = cuda_context->cuda_runtime_context.grid_dim.XYZ();

  int sm_count = cuda_context->cuda_device_context.sm_count;
  std::stringstream ss;
  ss << mononn_engine::helpers::string_format(
            "__launch_bounds__(%d, %d)", block_size,
            (block_count + sm_count - 1) / sm_count)
     << "\n";
  ss << "__global__"
     << "\n";
  ss << "void onefuser_kernel("
     << "\n";

  bool first = true;
  for (auto const& buffer_name :
       BufferManager::get_buffered_nodes_in_global()) {
    if (!first) ss << ",\n";
    first = false;
    ss << mononn_engine::helpers::string_format("void * __restrict__ %s",
                                                buffer_name.c_str());
  }

  ss << "\n";
  ss << ") {";
  ss << "\\\\Begin kernel\n";

  graph->wave_front_order([&](std::shared_ptr<ClusterOp> node) -> void {
    std::string node_name = node->get_name();
    ss << node->generate_cluster_code();

    for (auto const& edge : graph->get_node_output_edges(node_name)) {
      if (edge->need_sync()) {
        ss << edge->get_sync()->to_string();
      }
    }
  });

  ss << "\\\\End kernel\n";
  ss << "}\n";

  cuda_program.append(ss.str());

  return cuda_program;
}

void ClusteredGraphCodegen::initialize_buffer_manager(
    std::shared_ptr<ClusteredGraph> graph) {
  BufferManager::reset();

  graph->wave_front_order([&](std::shared_ptr<ClusterOp> op) -> void {
    for (auto const& node_name : op->get_graph()->get_input_nodes()) {
      if (!BufferManager::is_var_in_global(node_name)) {
        BufferManager::buffer_in_global(node_name);
      }
    }
  });

  for (auto const& cluster_node_name : graph->get_output_nodes()) {
    std::shared_ptr<ClusterOp> cluster_node =
        graph->get_node(cluster_node_name);
    for (auto const& node_name :
         cluster_node->get_graph()->get_output_nodes()) {
      BufferManager::buffer_in_global(node_name);
    }
  }

  std::string info = "Variable bufferred in global";
  for (auto const& node_name : BufferManager::get_buffered_nodes_in_global()) {
    info += "\t\t";
    info += node_name;
    info += "\n";
  }

  LOG(INFO) << info;
}

void ClusteredGraphCodegen::synchronization_analysis(
    std::shared_ptr<ClusteredGraph> graph) {
  graph->wave_front_order([&](std::shared_ptr<ClusterOp> op) -> void {
    std::string node_name = op->get_name();

    for (auto const& edge : graph->get_node_output_edges(node_name)) {
      std::shared_ptr<Synchronization> sync =
          std::make_shared<Synchronization>(Synchronization::Global);
      edge->set_sync(sync);  // all global sync for now;
    }
  });
}
}  // namespace codegen
}  // namespace mononn_engine