#pragma once
#include <memory>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include "tensorflow/mononn_extra/proto/graph_specification.pb.h"
#include "tensorflow/mononn_extra/proto/gemm_specification.pb.h"
#include "tensorflow/mononn_extra/proto/cuda_context.pb.h"
#include "mononn_engine/tuning/profiler/parallel_profiling_queue.h"
#include "mononn_engine/core/graph/graph.h"

namespace mononn_engine {
namespace tuning {
    class GraphCachedTuner {
    public:
        using Graph = mononn_engine::core::graph::Graph;
        using GraphSpecification = tensorflow::mononn_extra::proto::GraphSpecification;
        using ProfilingResult = mononn_engine::tuning::profiler::ProfilingResult;
        using Op = mononn_engine::core::op::Op;

        GraphCachedTuner(int _num_process, std::shared_ptr<Graph> &_graph)
                : profiling_queue(_num_process), graph(_graph) {}

        std::unique_ptr<GraphSpecification> get_optimal_spec(std::vector<GraphSpecification const *> candidate_spec_list);

    private:
        void check_profiling_result(const ProfilingResult &result) const;
        static std::string context_to_str(tensorflow::mononn_extra::proto::CUDAContext const *cuda_context);
        std::string get_gemm_or_conv_problem_hash(const Op *gemm_op) const;
        void dump_performance_report(const std::unordered_map<std::string, std::unordered_map<std::string, float>> &report) const;
        profiler::ParallelProfilingQueue profiling_queue;
        std::shared_ptr<Graph> graph;
    };
}
}