#include <algorithm>

#include "mononn_engine/optimization/pass_runner.h"
#include "tensorflow/core/platform/logging.h"
#include "mononn_engine/config/config.h"

namespace mononn_engine {
namespace optimization {
    using Config = mononn_engine::config::Config;

    bool PassRunner::run(Graph *graph, std::shared_ptr<CUDAContext> cuda_context) {
        if (std::find(Config::get()->optimization_disabled_pass.begin(),
                      Config::get()->optimization_disabled_pass.end(),
                      this->pass->name()) != Config::get()->optimization_disabled_pass.end()) {
            LOG(INFO) << "Optimization pass " << this->pass->name() << " disabled.";
            ++this->run_cnt;
            this->succeed = false;
            return this->succeed;
        }

        LOG(INFO) << "Run optimization pass: " << this->pass->name();

        this->succeed = this->pass->run(graph, cuda_context);
        ++this->run_cnt;

        if (!this->succeed) LOG(INFO) << "Pass: " << this->pass->name() << " do not have identified pattern";

        return this->succeed;
    }
}
}
