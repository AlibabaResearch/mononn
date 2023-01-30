#pragma once
#include <memory>
#include <string>
#include <vector>

#include "mononn_engine/core/context/cuda_context.h"
#include "mononn_engine/core/graph/graph.h"

namespace mononn_engine {
namespace optimization {
// Optimization pass that operate graph
class GraphPass {
 public:
  using Graph = mononn_engine::core::graph::Graph;
  using CUDAContext = mononn_engine::core::context::CUDAContext;

  virtual std::string name() const = 0;
  virtual bool run(Graph* graph, std::shared_ptr<CUDAContext> cuda_context) = 0;

 private:
};
}  // namespace optimization
}  // namespace mononn_engine