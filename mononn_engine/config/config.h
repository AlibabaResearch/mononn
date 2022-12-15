#pragma once

#include <memory>
#include <vector>

namespace mononn_engine {
namespace config {
    struct Config {
    public:
        static Config* get();

        bool print_hlo_text;
        std::string output_dir;
        std::string hlo_file;
        std::string onefuser_buffer_name;
        std::string graph_spec_compiler_path;
        std::vector<int> gpus;
        std::vector<std::string> feeds;
        std::vector<std::string> fetches;
        std::vector<std::string> input_data_files;
        std::vector<std::string> optimization_disabled_pass;
        std::vector<int> candidate_ilp_factor;
        bool save_candidate_specification;
        bool run_expensive_verification;
        std::vector<std::string> host_codegen_disabled_pass;
        bool gemm_tensor_op_enabled;
        bool gemm_simt_enabled;
        bool faster_tuning;
        bool fastest_tuning;
        bool use_cached_tuning_result;

        std::string hlo_module_proto_temp_path;
        std::string hlo_module_proto_temp_file;
        std::string dump_text_hlo_dir;
        std::string onefuser_home;
        std::string smem_reduction_cache_name;

        std::string get_hlo_module_proto_temp_file();
    private:

        static std::unique_ptr<Config> config;
    };
}
}

