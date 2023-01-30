#include "mononn_engine/config/config.h"

#include <cstdlib>
#include <experimental/filesystem>
#include <iostream>

namespace mononn_engine {
namespace config {
std::unique_ptr<Config> Config::config = nullptr;
namespace fs = std::experimental::filesystem;

std::vector<std::string> str_split(const std::string& str,
                                   const std::string& delimiter) {
  std::string s = str;
  std::vector<std::string> tokens;

  size_t pos = 0;
  std::string token;
  while ((pos = s.find(delimiter)) != std::string::npos) {
    token = s.substr(0, pos);
    tokens.push_back(token);
    s.erase(0, pos + delimiter.length());
  }

  if (!s.empty()) {
    tokens.push_back(s);
  }

  return tokens;
}

Config* Config::get() {
  if (config == nullptr) {
    config = std::make_unique<Config>();
    config->print_hlo_text = true;
    config->onefuser_buffer_name = "onefuser_memory_buffer";

    config->onefuser_home = fs::current_path();
    if (std::getenv("ONEFUSER_HOME")) {
      config->onefuser_home = std::getenv("ONEFUSER_HOME");
    }

    if (std::getenv("ONEFUSER_GRAPH_SPEC_COMPILER_PATH")) {
      config->graph_spec_compiler_path =
          std::getenv("ONEFUSER_GRAPH_SPEC_COMPILER_PATH");
    } else {
      config->graph_spec_compiler_path =
          config->onefuser_home +
          "/bazel-bin/tensorflow/onefuser/codegen/"
          "graph_specification_codegen_main";
    }

    // if (!fs::exists(config->graph_spec_compiler_path)) {
    //     std::cerr << "At " << __FILE__ << ":" << __LINE__ << " " <<
    //     config->graph_spec_compiler_path << " not exists."; abort();
    // }

    config->save_candidate_specification = true;
    config->gpus = {0};
    config->run_expensive_verification = true;
    config->gemm_tensor_op_enabled = true;
    config->gemm_simt_enabled = true;
    config->faster_tuning = false;
    config->fastest_tuning = false;
    config->use_cached_tuning_result = true;

    config->smem_reduction_cache_name = "s_cache_reduction";

    config->candidate_ilp_factor = {1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32};

    if (getenv("MONONN_OPTIMIZATION_DISABLED_PASS")) {
      std::string pass_list = getenv("MONONN_OPTIMIZATION_DISABLED_PASS");

      config->optimization_disabled_pass = str_split(pass_list, ",");
    }
  }

  return config.get();
}
}  // namespace config
}  // namespace mononn_engine