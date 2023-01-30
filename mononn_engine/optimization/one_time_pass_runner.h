#pragma once
#include "mononn_engine/optimization/pass_runner.h"

namespace mononn_engine {
namespace optimization {

class OneTimePassRunner : public PassRunner {
 public:
  OneTimePassRunner(std::unique_ptr<GraphPass> _pass)
      : PassRunner(std::move(_pass)) {}
  bool can_run() const override;

 private:
};

}  // namespace optimization
}  // namespace mononn_engine
