#pragma once
#include <memory>

#include "mononn_engine/core/graph/graph.h"
#include "mononn_engine/optimization/graph_pass.h"

namespace mononn_engine {
namespace optimization {
class PassRunner {
 public:
  using Graph = mononn_engine::core::graph::Graph;
  using CUDAContext = mononn_engine::core::context::CUDAContext;

  PassRunner(std::unique_ptr<GraphPass> _pass) : pass(std::move(_pass)) {}
  virtual bool can_run() const = 0;
  bool run(Graph* graph, std::shared_ptr<CUDAContext> cuda_ontext);

 protected:
  std::unique_ptr<GraphPass> pass;
  int run_cnt = 0;
  bool succeed = true;
};
}  // namespace optimization
}  // namespace mononn_engine
