#pragma once

#include "mononn_engine/core/context/cuda_context.h"
#include "mononn_engine/core/op/cluster_op.h"

namespace mononn_engine {
namespace codegen {
class ClusterCodegen {
 public:
  using ClusterOp = mononn_engine::core::op::ClusterOp;
  using CUDAContext = mononn_engine::core::context::CUDAContext;
  static void setup_codegen(std::shared_ptr<const CUDAContext> cuda_context,
                            std::shared_ptr<ClusterOp> cluster_op);
  static std::string generate(std::shared_ptr<const CUDAContext> cuda_context,
                              std::shared_ptr<const ClusterOp> cluster_op);
  static std::string generate_function_declaration(
      std::shared_ptr<const CUDAContext> cuda_context,
      std::shared_ptr<const ClusterOp> cluster_op);
  static std::string generate_function_definition(
      std::shared_ptr<const CUDAContext> cuda_context,
      std::shared_ptr<const ClusterOp> cluster_op);

 private:
};
}  // namespace codegen
}  // namespace mononn_engine