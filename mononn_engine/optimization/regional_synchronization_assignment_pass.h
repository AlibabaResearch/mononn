#pragma once
#include <memory>
#include <string>

#include "mononn_engine/optimization/graph_pass.h"

namespace mononn_engine {
namespace optimization {
class RegionalSynchronizationAssignmentPass : public GraphPass {
 public:
  std::string name() const override;
  bool run(Graph* graph, std::shared_ptr<CUDAContext> cuda_context) override;
};
}  // namespace optimization
}  // namespace mononn_engine
