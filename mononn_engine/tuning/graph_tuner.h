#pragma once
#include <memory>
#include <vector>
#include <unordered_set>
#include "tensorflow/mononn_extra/proto/graph_specification.pb.h"
#include "tensorflow/mononn_extra/proto/gemm_specification.pb.h"
#include "tensorflow/mononn_extra/proto/cuda_context.pb.h"
#include "mononn_engine/tuning/profiler/parallel_profiling_queue.h"
#include "mononn_engine/core/graph/graph.h"

namespace mononn_engine {
namespace tuning {
    class GraphTuner {
    public:
        using Graph = mononn_engine::core::graph::Graph;
        using GraphSpecification = tensorflow::mononn_extra::proto::GraphSpecification;
        using ProfilingResult = mononn_engine::tuning::profiler::ProfilingResult;
        using Op = mononn_engine::core::op::Op;

        GraphTuner(int _num_process, std::shared_ptr<Graph> &_graph)
            : profiling_queue(_num_process), graph(_graph) {}

        std::unique_ptr<GraphSpecification> get_optimal_spec(std::vector<GraphSpecification const *> candidate_spec_list);

    private:
        void check_profiling_result(const ProfilingResult &result) const;
        std::vector<std::unique_ptr<GraphSpecification>> get_optimal_spec_for_each_cuda_context(std::vector<ProfilingResult> &profiling_result, std::vector<GraphSpecification const *> candidate_spec_list);
        std::unique_ptr<GraphSpecification> get_optimal_spec_for_cuda_context(std::vector<ProfilingResult> &profiling_result, std::vector<GraphSpecification const *> candidate_spec_list);
        std::unique_ptr<GraphSpecification> get_optimal_spec_for_node(std::vector<ProfilingResult> &profiling_result, std::vector<GraphSpecification const *> candidate_spec_list);
        static std::string context_to_str(tensorflow::mononn_extra::proto::CUDAContext const *cuda_context);
        profiler::ParallelProfilingQueue profiling_queue;

        std::shared_ptr<Graph> graph;
    };
}
}