#pragma once
#include <memory>
#include <string>
#include <vector>

#include "mononn_engine/core/graph/clustered_graph.h"

namespace mononn_engine {
namespace optimization {
// Optimization pass that operate on clustered graph
class ClusteredGraphPass {
 public:
  using ClusteredGraph = mononn_engine::core::graph::ClusteredGraph;
  virtual std::string name() const = 0;
  virtual bool run(std::shared_ptr<ClusteredGraph> graph) = 0;

 private:
};
}  // namespace optimization
}  // namespace mononn_engine