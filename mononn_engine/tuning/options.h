// Copyright 2023 The MonoNN Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
