#include <thread>
#include "mononn_engine/codegen/compilation_threadpool.h"
#include "mononn_engine/frontend/mononn_module.h"
#include "mononn_engine/parser/ir_parser_fused.h"
#include "mononn_engine/optimization/optimization_runner.h"
#include "mononn_engine/helpers/env_variable.h"
#include "mononn_engine/core/common/compile_output_type.h"

namespace mononn_engine {
namespace codegen {
    using MonoNNModule = mononn_engine::frontend::MonoNNModule;
    using OptimizationRunner = mononn_engine::optimization::OptimizationRunner;
    using IRParserFused = mononn_engine::parser::IRParserFused;
    using EnvVar = mononn_engine::helpers::EnvVar;
    using CompileOutputType = mononn_engine::core::common::CompileOutputType::Type;

    CompilationThreadpool::CompilationThreadpool(int n_threads) : thread_pool(n_threads) {
        LOG(INFO) << "CompilationThreadpool size: " << n_threads;
    }

    CompilationThreadpool::CompilationThreadpool() : 
        CompilationThreadpool(EnvVar::defined("TF_MONONN_THREADPOOL_SIZE") ? 
            std::stoi(EnvVar::get("TF_MONONN_THREADPOOL_SIZE")) : std::thread::hardware_concurrency()) {}

    void CompilationThreadpool::post(
        const xla::HloModule *hlo_module, 
        std::unique_ptr<Graph> graph, 
        TuningSpecId tuning_spec_id,
        std::unique_ptr<GraphSpecification> tuning_spec, 
        const std::string &kernel_name,
        const std::vector<xla::BufferAllocation> *allocation_list,
        const HloAliasAnalysis *alias_analysis,
        ConcurrentQueue<Result> &finish_queue,
        CompileDebugOption compile_debug_option) {

        this->thread_pool.enqueue([compile_debug_option, &finish_queue, hlo_module, graph = std::move(graph), tuning_spec_id, tuning_spec = std::move(tuning_spec), kernel_name, allocation_list, alias_analysis] () mutable {

            std::unique_ptr<MonoNNModule> mononn_module 
                = std::make_unique<MonoNNModule>(hlo_module, std::move(graph), std::move(tuning_spec), kernel_name, allocation_list, alias_analysis);
            
            // Assume have already finish pre impl assignment optimizations, which is time consuming and invariant across tuning specifications.
            // mononn_module->set_optimizations_have_done(OptimizationRunner::GRPOU_PRE_IMPL_ASSIGNMENT);

            mononn_module->generate_code();
            mononn_module->build_assembly(CompileOutputType::COMPILE_OUTPUT_TYPE_PTX);
            Result result;
            result.tuning_spec_id = tuning_spec_id;
            result.ptx = mononn_module->get_ptx();

            if (compile_debug_option & CompileDebugOption::COMPILATION_SAVE_SOURCE) {
                std::vector<std::string> file_list = mononn_module->get_cuda_program()->get_file_list();

                for (auto const &file : file_list) {
                    result.file_collection[file] = mononn_module->get_cuda_program()->file_ref(file).str();
                }
            }
            
            finish_queue.push(result);
        });
    }

    void CompilationThreadpool::post(
        const xla::HloModule *hlo_module, 
        TuningSpecId tuning_spec_id,
        std::unique_ptr<GraphSpecification> tuning_spec, 
        const std::string &kernel_name,
        const std::vector<xla::BufferAllocation> *allocation_list,
        const HloAliasAnalysis *alias_analysis,
        ConcurrentQueue<Result> &finish_queue,
        CompileDebugOption compile_debug_option) {

        this->thread_pool.enqueue([compile_debug_option, &finish_queue, hlo_module, tuning_spec_id, tuning_spec = std::move(tuning_spec), kernel_name, allocation_list, alias_analysis] () mutable {

            std::unique_ptr<Graph> graph = IRParserFused::from_hlo_module_unique(hlo_module);
            std::unique_ptr<MonoNNModule> mononn_module 
                = std::make_unique<MonoNNModule>(hlo_module, std::move(graph), std::move(tuning_spec), kernel_name, allocation_list, alias_analysis);
            
            mononn_module->generate_code();
            mononn_module->build_assembly(CompileOutputType::COMPILE_OUTPUT_TYPE_PTX);
            Result result;
            result.tuning_spec_id = tuning_spec_id;
            result.ptx = mononn_module->get_ptx();

            if (compile_debug_option & CompileDebugOption::COMPILATION_SAVE_SOURCE) {
                std::vector<std::string> file_list = mononn_module->get_cuda_program()->get_file_list();

                for (auto const &file : file_list) {
                    result.file_collection[file] = mononn_module->get_cuda_program()->file_ref(file).str();
                }
            }
            
            finish_queue.push(result);
        });
    }

    size_t CompilationThreadpool::num_remaining_tasks() {
        return this->thread_pool.num_remaining_tasks();
    }
}
}