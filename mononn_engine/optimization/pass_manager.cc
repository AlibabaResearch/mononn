#include "mononn_engine/optimization/pass_manager.h"
#include "mononn_engine/config/config.h"

namespace mononn_engine {
namespace optimization {
    using Config = mononn_engine::config::Config;

    void PassManager::add_runner(std::unique_ptr<PassRunner> runner) {
        this->runner_list.push_back(std::move(runner));
    }

    void PassManager::execute(Graph *graph) {
        LOG(INFO) << "Begin pass manager optimization";
        for (auto const &runner : this->runner_list) {
            while (runner->can_run()) {
                runner->run(graph, this->cuda_context);

                if (Config::get()->run_expensive_verification) {
                    std::vector<std::string> node_list;
                    if (!graph->is_acyclic(node_list, true)) {
                        LOG(FATAL) << "Detected cycle in graph: " << mononn_engine::helpers::join(" ", node_list);
                    }
                }
            }
        }
        LOG(INFO) << "End pass manager optimization";
    }

    void PassManager::clear_runner() {
        this->runner_list.clear();
    }
}
}