#include "mononn_engine/optimization/one_time_pass_runner.h"

namespace mononn_engine {
namespace optimization {
    bool OneTimePassRunner::can_run() const {
        return this->run_cnt == 0;
    }
}
}