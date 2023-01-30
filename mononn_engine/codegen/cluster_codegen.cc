#include "mononn_engine/codegen/cluster_codegen.h"

#include <sstream>

#include "mononn_engine/core/context/index_tracer.h"
#include "mononn_engine/core/gpu/buffer_manager.h"
#include "mononn_engine/core/gpu/defined.h"
#include "mononn_engine/core/gpu/functor.h"
#include "mononn_engine/core/op/parameter.h"
#include "mononn_engine/core/semantic/function_invocation.h"
#include "mononn_engine/helpers/string_helpers.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace codegen {
using Schedule = mononn_engine::core::schedule::Schedule;
using IndexTracer = mononn_engine::core::context::IndexTracer;
using FunctionInvocation = mononn_engine::core::semantic::FunctionInvocation;
using Functor = mononn_engine::core::gpu::Functor;
using Op = mononn_engine::core::op::Op;
using OpType = mononn_engine::core::op::OpType;
using Parameter = mononn_engine::core::op::Parameter;
using BufferManager = mononn_engine::core::gpu::BufferManager;
using CUDADefined = mononn_engine::core::gpu::CUDADefined;

std::string ClusterCodegen::generate(
    std::shared_ptr<const CUDAContext> cuda_context,
    std::shared_ptr<const ClusterOp> cluster_op) {
  return cluster_op->generate_cluster_invocation();
}

void ClusterCodegen::setup_codegen(
    std::shared_ptr<const CUDAContext> cuda_context,
    std::shared_ptr<ClusterOp> cluster_op) {
  cluster_op->setup_codegen();
}

std::string ClusterCodegen::generate_function_declaration(
    std::shared_ptr<const CUDAContext> cuda_context,
    std::shared_ptr<const ClusterOp> cluster_op) {
  std::stringstream ss;
  ss << "__device__ __forceinline__"
     << "\n";
  ss << "void " << cluster_op->get_name() << "_computation("
     << "\n";

  for (auto const& node_name : cluster_op->get_graph()->get_input_nodes()) {
    ss << "void *" << BufferManager::get_buffer_name(node_name) + "_input";
    ss << ","
       << "\n";
  }

  int output_node_count =
      (int)cluster_op->get_graph()->get_output_nodes().size();
  for (int idx = 0; idx < output_node_count; ++idx) {
    std::string node_name = cluster_op->get_graph()->get_output_node(idx);
    std::shared_ptr<const Op> node =
        cluster_op->get_graph()->get_node(node_name);

    if (node->get_output_specs_count() > 1) {
      if (node->get_type() != OpType::reduce) {
        LOG(FATAL) << "Reduce node expected, however " << node->get_name()
                   << " has type " << node->get_type().to_string();
      }

      for (int tuple_index = 0; tuple_index < node->get_output_specs_count();
           ++tuple_index) {
        ss << "void *"
           << BufferManager::get_buffer_name(node_name) + "_tuple_index_" +
                  std::to_string(tuple_index) + "_output";

        if (tuple_index != node->get_output_specs_count() - 1) {
          ss << ",\n";
        }
      }
    } else {
      ss << "void *" << BufferManager::get_buffer_name(node_name) + "_output";
    }

    if (idx == output_node_count - 1)
      ss << "\n";
    else
      ss << ",\n";
  }

  ss << ")";

  return ss.str();
}

std::string ClusterCodegen::generate_function_definition(
    std::shared_ptr<const CUDAContext> cuda_context,
    std::shared_ptr<const ClusterOp> cluster_op) {
  std::stringstream ss;

  ss << ClusterCodegen::generate_function_declaration(cuda_context, cluster_op);

  ss << "{"
     << "\n";
  ss << CUDADefined::initialize(cuda_context.get()) << "\n";
  ss << "// Define functor begin\n\n";
  // ss << Functor::get_all_functors_definition() << "\n";
  ss << "// Define functor end\n\n";
  ss << cluster_op->generate_cluster_code();

  ss << "}\n\n";

  return ss.str();
}
}  // namespace codegen
}  // namespace mononn_engine