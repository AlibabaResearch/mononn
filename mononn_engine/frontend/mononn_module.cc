#include "mononn_engine/frontend/mononn_module.h"
#include "mononn_engine/parser/ir_parser_fused.h"
#include "mononn_engine/optimization/optimization_runner.h"
#include "mononn_engine/helpers/helpers.h"
#include "mononn_engine/codegen/graph_codegen.h"
#include "mononn_engine/tuning/profiler/subprocess.h"
#include "mononn_engine/core/common/compile_output_type.h"

namespace mononn_engine {
namespace frontend {
    using Graph = mononn_engine::core::graph::Graph;
    using CUDAContext = mononn_engine::core::context::CUDAContext;
    using OptimizationRunner = mononn_engine::optimization::OptimizationRunner;
    using GraphCodegen = mononn_engine::codegen::GraphCodegen;
    using SubProcess = mononn_engine::tuning::profiler::SubProcess;
    using EnvVar = mononn_engine::helpers::EnvVar;
    
    void MonoNNModule::set_tuning_spec(std::unique_ptr<GraphSpecification> _tuning_spec) {
        this->tuning_spec = std::move(_tuning_spec);
    }

    const GraphSpecification *MonoNNModule::get_tuning_spec() const {
        return this->tuning_spec.get();
    }

    void MonoNNModule::generate_code() {
        if (!this->graph) {
            this->graph = mononn_engine::parser::IRParserFused::from_hlo_module_unique(this->hlo_module);
        }

        std::shared_ptr<CUDAContext> cuda_context = std::make_shared<CUDAContext>();
        cuda_context->ParseFromProto(&this->tuning_spec->cuda_context());

        if (!(this->optimizations_have_done & OptimizationRunner::GRPOU_PRE_IMPL_ASSIGNMENT)) {
            OptimizationRunner::run_group_pre_impl_assignment(graph.get(), cuda_context);
        }
        
        OptimizationRunner::run_group_impl_assignment(graph.get(), cuda_context, this->tuning_spec.get());
        OptimizationRunner::run_group_impl_optimization(graph.get(), cuda_context, this->tuning_spec.get());
        OptimizationRunner::run_group_buffer_assignment(graph.get(), cuda_context);

        std::unordered_set<std::string> codegen_reject_list;

        if (EnvVar::defined("MONONN_CODEGEN_ALLOW_LIST_OVERRIDE")) {
            std::string allowed_node_name = EnvVar::get("MONONN_CODEGEN_ALLOW_LIST_OVERRIDE");
            bool found_allowed_node = false;

            for (auto const &node_name : this->tuning_spec->codegen_allow_list()) {
                if (node_name != allowed_node_name) {
                    codegen_reject_list.insert(node_name);
                } else {
                    found_allowed_node = true;
                }
            }

            for (auto const &node_name : this->tuning_spec->codegen_reject_list()) {
                if (node_name != allowed_node_name) {
                    codegen_reject_list.insert(node_name);
                } else {
                    found_allowed_node = true;
                }
            }

            if (!found_allowed_node) {
                LOG(FATAL) << "Node " << allowed_node_name << " not found.";
            }
        } else {
            codegen_reject_list.insert(this->tuning_spec->codegen_reject_list().begin(), this->tuning_spec->codegen_reject_list().end());
        }
        
        std::vector<std::string> argument_list;

        argument_list.resize(this->hlo_module->entry_computation()->num_parameters());

        int liveout_buffer_index = 0;
        uint64_t temp_buffer = 0;

        std::unordered_map<uint64_t, std::string> allocation_ptr_to_buffer_name;

        for (auto const &allocation : *this->allocation_list) {
            if (allocation.is_entry_computation_parameter()) {
                if (allocation.assigned_buffers().size() != 1) {
                    LOG(FATAL) << "Parameter buffer " << allocation.parameter_number() << " should have exactly logical buffer, got " << allocation.assigned_buffers().size()
                            << "\n" << allocation.ToString();
                }

                std::string node_name = allocation.assigned_buffers().begin()->first->defining_instruction()->name();
                std::string buffer_name = mononn_engine::helpers::get_canonicalized_node_name(node_name);
                argument_list[allocation.parameter_number()] = buffer_name;
                allocation_ptr_to_buffer_name[reinterpret_cast<uint64_t>(&allocation)] = buffer_name;
            }

            if (allocation.maybe_live_out()) {
                if (allocation.is_tuple()) {
                    // We do not need to pass tuple buffer (typically last node in the entry computation).
                    // The tuple info is not in buffer_allocations in GetMlirAllocationInfo neither.
                    continue; 
                }

                std::string buffer_name = "liveout_buffer_" + std::to_string(liveout_buffer_index++);
                argument_list.push_back(buffer_name);
                allocation_ptr_to_buffer_name[reinterpret_cast<uint64_t>(&allocation)] = buffer_name;
            }

            if (allocation.IsPreallocatedTempBuffer()) {
                temp_buffer = reinterpret_cast<uint64_t>(&allocation);
            }
        }

        if (temp_buffer) { 
            allocation_ptr_to_buffer_name[temp_buffer] = "temp_buffer";
            argument_list.push_back("temp_buffer");
        }

        GraphCodegen::Params params {
            cuda_context,
            graph.get(),
            codegen_reject_list,
            kernel_name,
            argument_list,
            false,
            false,
            GraphCodegen::MONONN_BUFFER_MANAGEMENT_TF_XLA,
            this->allocation_list,
            allocation_ptr_to_buffer_name,
            this->hlo_module,
            this->alias_analysis
        };

        this->cuda_program = 
            std::move(GraphCodegen::generate(params));
    }

    void MonoNNModule::build_assembly(CompileOutputType type) {
        if (!this->cuda_program) {
            LOG(FATAL) << "CUDA program must be generated before build cubin.";
        }
        
        helpers::TempDirectoryRAII tmp_dir_guard(mononn_engine::helpers::Directory::get_mononn_new_temp_dir());

        cuda_program->generate(tmp_dir_guard.get_dir_name(), type);

        std::string build_cmd = "cd " + tmp_dir_guard.get_dir_name() + " && make";
        SubProcess build_process(build_cmd, {"2>&1"});

        build_process.start();
        build_process.wait();

        if (build_process.get_return_code() != 0) {
            LOG(FATAL) << "Generated program build failed: " << build_process.get_output() 
                << "\nDirectory: " << tmp_dir_guard.get_dir_name();
        }

        if (type == CompileOutputType::COMPILE_OUTPUT_TYPE_CUBIN) {
            std::string cubin_file_name = mononn_engine::helpers::Path::join(tmp_dir_guard.get_dir_name(), "mononn.cubin");
            this->cubin = mononn_engine::helpers::File::read_as_binary(cubin_file_name);
        } else if (type == CompileOutputType::COMPILE_OUTPUT_TYPE_PTX) {
            std::string ptx_file_name = mononn_engine::helpers::Path::join(tmp_dir_guard.get_dir_name(), "mononn.ptx");
            this->ptx = mononn_engine::helpers::File::read_as_string(ptx_file_name);
        } else {
            LOG(FATAL) << "Unsupported type: " << type;
        }
    }

    bool MonoNNModule::has_cubin() const {
        return !this->cubin.empty();
    }

    bool MonoNNModule::has_ptx() const {
        return !this->ptx.empty();
    }

    void MonoNNModule::set_cubin(const std::vector<uint8_t> &_cubin) {
        this->cubin = _cubin;
    }

    void MonoNNModule::set_ptx(const std::string &_ptx) {
        this->ptx = _ptx;
    }

    const std::vector<uint8_t>& MonoNNModule::get_cubin() const {
        if (this->cubin.empty()) {
            LOG(FATAL) << "Cubin does not exists.";
        }

        return this->cubin;
    }

    const std::string& MonoNNModule::get_ptx() const {
        if (this->ptx.empty()) {
            LOG(FATAL) << "PTX does not exists.";
        }

        return this->ptx;
    }

    void MonoNNModule::set_optimizations_have_done(int _optimizations_have_done) {
        this->optimizations_have_done |= _optimizations_have_done;
    }

    const CUDAProgram * MonoNNModule::get_cuda_program() const {
        return this->cuda_program.get();
    }
}
}