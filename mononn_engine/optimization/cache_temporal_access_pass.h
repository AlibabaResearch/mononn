#pragma once

#include "mononn_engine/optimization/graph_pass.h"

namespace mononn_engine {
namespace optimization {
class CacheTemporalAccessPass : public GraphPass {
 public:
  std::string name() const override;
  bool run(Graph* graph, std::shared_ptr<CUDAContext> cuda_context) override;

 private:
};
}  // namespace optimization
}  // namespace mononn_engine
