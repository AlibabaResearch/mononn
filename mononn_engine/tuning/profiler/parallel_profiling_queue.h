#pragma once

#include <mutex>
#include <future>

#include "mononn_engine/tuning/profiler/thread_pool.h"
#include "mononn_engine/tuning/profiler/profiling_result.h"
#include "tensorflow/mononn_extra/proto/graph_specification.pb.h"


namespace mononn_engine {
namespace tuning {
namespace profiler {
    class ParallelProfilingQueue {
    public:
        struct NCUResult {
            float time_in_us;
        };

        ParallelProfilingQueue(int n_threads) : thread_pool(n_threads) {}

        using GraphSpecification = tensorflow::mononn_extra::proto::GraphSpecification;
        // Non blocking post
        std::future<ProfilingResult> post(GraphSpecification const *graph_spec, std::vector<std::string> host_codegen_disabled_pass = {}, std::vector<std::string> optimization_disabled_pass = {});
    private:
        ThreadPool thread_pool;

        std::mutex thread_mutex;
        std::mutex profile_mutex[8];

        NCUResult parse_ncu_result(std::string str) const;
        NCUResult parse_console_result(std::string str) const;
    };
}
}
}
