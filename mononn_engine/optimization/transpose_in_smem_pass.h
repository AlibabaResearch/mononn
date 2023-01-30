#pragma once
#include "mononn_engine/optimization/graph_pass.h"
namespace mononn_engine {
namespace optimization {
class TransposeInSmemPass : public GraphPass {
  std::string name() const override;
  bool run(Graph* graph, std::shared_ptr<CUDAContext> cuda_context) override;
};
}  // namespace optimization
}  // namespace mononn_engine
