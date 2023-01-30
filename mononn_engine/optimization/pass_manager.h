#pragma once
#include <memory>

#include "mononn_engine/core/context/cuda_context.h"
#include "mononn_engine/optimization/pass_runner.h"

namespace mononn_engine {
namespace optimization {
using Graph = mononn_engine::core::graph::Graph;
using CUDAContext = mononn_engine::core::context::CUDAContext;

class PassManager {
 public:
  explicit PassManager(std::shared_ptr<CUDAContext> _cuda_context)
      : cuda_context(_cuda_context) {}

  void add_runner(std::unique_ptr<PassRunner> runner);
  void execute(Graph* graph);
  void clear_runner();

 private:
  std::vector<std::unique_ptr<PassRunner>> runner_list;
  std::shared_ptr<CUDAContext> cuda_context;
};
}  // namespace optimization
}  // namespace mononn_engine
