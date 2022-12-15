#include "mononn_engine/optimization/run_greedy_pass_runner.h"

namespace mononn_engine {
namespace optimization {
    bool RunGreedyPassRunner::can_run() const {
        return this->succeed;
    }
}
}