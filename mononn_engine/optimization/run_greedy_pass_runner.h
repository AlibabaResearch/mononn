#include <memory>

#include "mononn_engine/optimization/pass_runner.h"

namespace mononn_engine {
namespace optimization {
    class RunGreedyPassRunner : public PassRunner {
    public:
        RunGreedyPassRunner(std::unique_ptr<GraphPass> _pass) : PassRunner(std::move(_pass)) {}
        bool can_run() const override;
    private:
    };
}
}

