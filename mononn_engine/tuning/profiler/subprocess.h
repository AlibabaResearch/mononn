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
namespace profiler {
class SubProcess {
 public:
  SubProcess(){};
  SubProcess(std::string const& _cmd) : cmd(_cmd) {}
  SubProcess(std::string const& _cmd, std::vector<std::string> const& _args)
      : cmd(_cmd), args(_args) {}

  void start();
  void wait();

  int get_return_code() const;
  const std::string& get_output() const;

 private:
  std::string cmd;
  std::vector<std::string> args;
  FILE* fp;
  int return_code;
  std::string output;
};
}  // namespace profiler
}  // namespace tuning
}  // namespace mononn_engine
