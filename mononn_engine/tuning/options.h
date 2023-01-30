#pragma once

#include <string>
#include <vector>

namespace mononn_engine {
namespace tuning {
struct Options {
  std::string input_file;
  int num_threads;
  std::string output_dir;
  std::string dump_text_hlo_dir;
  std::vector<std::string> input_data_files;
  std::vector<int> gpus;
  bool automatic_mixed_precision;
  bool faster_tuning;
  bool fastest_tuning;
  bool use_cached_tuning_result;
  std::vector<std::string> feeds;
  std::vector<std::string> fetches;
  std::vector<std::string> optimization_disabled_pass;
};
}  // namespace tuning
}  // namespace mononn_engine
