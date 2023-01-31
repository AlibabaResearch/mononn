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
namespace parser {
class HloModuleDumper {
 public:
  HloModuleDumper(const std::string& _frozen_pb_file,
                  const bool& _auto_mixed_precision,
                  const std::vector<std::string>& _feeds,
                  const std::vector<std::string>& _input_files,
                  const std::vector<std::string>& _fetches)
      : frozen_pb_file(_frozen_pb_file),
        auto_mixed_precision(_auto_mixed_precision),
        feeds(_feeds),
        input_files(_input_files),
        fetches(_fetches) {}

  void dump(std::string binary_path, std::string text_path = "");

 private:
  std::string frozen_pb_file;
  bool auto_mixed_precision;
  std::vector<std::string> feeds;
  std::vector<std::string> input_files;
  std::vector<std::string> fetches;
};
}  // namespace parser
}  // namespace mononn_engine