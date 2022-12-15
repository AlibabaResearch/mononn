#pragma once

#include <vector>
#include <future>
#include "mononn_engine/tuning/profiler/thread_pool.h"
#include "mononn_engine/core/common/concurrent_queue.h"
#include "mononn_engine/core/graph/graph.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/mononn_extra/proto/graph_specification.pb.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"

namespace mononn_engine {
namespace codegen {
    using BufferAllocation = xla::BufferAllocation;
    using HloAliasAnalysis  = xla::HloAliasAnalysis;
    using ThreadPool = mononn_engine::tuning::profiler::ThreadPool;
    using Graph = mononn_engine::core::graph::Graph;
    using GraphSpecification = tensorflow::mononn_extra::proto::GraphSpecification;
    template<typename T>
    using ConcurrentQueue = mononn_engine::core::common::ConcurrentQueue<T>;
    using TuningSpecId = uint64_t;

    class CompilationThreadpool {
    public:
        struct Result {
            using FileName = std::string;
            using FileContent = std::string;
            using FileCollection = std::unordered_map<FileName, FileContent>;

            TuningSpecId tuning_spec_id;
            std::string ptx;
            std::vector<uint8_t> cubin;
            FileCollection file_collection;
        };

        enum CompileDebugOption {
            NONE = 0,
            COMPILATION_SAVE_SOURCE = 1 << 0
        };

        CompilationThreadpool();
        CompilationThreadpool(int n_threads);
        void post(
            const xla::HloModule *hlo_module, 
            std::unique_ptr<Graph> graph, 
            TuningSpecId tuning_spec_id,
            std::unique_ptr<GraphSpecification> tuning_spec, 
            const std::string &kernel_name,
            const std::vector<xla::BufferAllocation> *allocation_list,
            const HloAliasAnalysis *alias_analysis,
            ConcurrentQueue<Result> &finish_queue,
            CompileDebugOption compile_debug_option = CompileDebugOption::NONE);

        void post(
            const xla::HloModule *hlo_module, 
            TuningSpecId tuning_spec_id,
            std::unique_ptr<GraphSpecification> tuning_spec, 
            const std::string &kernel_name,
            const std::vector<xla::BufferAllocation> *allocation_list,
            const HloAliasAnalysis *alias_analysis,
            ConcurrentQueue<Result> &finish_queue,
            CompileDebugOption compile_debug_option = CompileDebugOption::NONE);

        template<typename F, typename ... Args>
        auto post_general(F &&f, Args &&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
            return std::move(this->thread_pool.enqueue(std::forward<F>(f), std::forward<Args>(args)...));
        }
        
        size_t num_remaining_tasks();
    private:
        ThreadPool thread_pool;
    };
};
}