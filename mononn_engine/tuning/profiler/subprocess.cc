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

#include "mononn_engine/tuning/profiler/subprocess.h"

#include <cstdlib>

#include <errno.h>
#include "mononn_engine/helpers/helpers.h"

namespace mononn_engine {
namespace tuning {
namespace profiler {
void SubProcess::start() {
  std::string command = this->cmd;
  for (auto const& arg : this->args) {
    command += " " + arg;
  }

  //        LOG(DEBUG) << "Run cmd: " << command;
  this->fp = popen(command.c_str(), "r");

  if (this->fp == nullptr) {
    LOG(FATAL) << "Subprocess: command " << command << " execution failed\n"
      << "Popen error: " << strerror(errno);
  }
}

void SubProcess::wait() {
  constexpr int BUF_SIZE = 2048;
  char buf[BUF_SIZE];
  while (fgets(buf, BUF_SIZE, this->fp) != nullptr) {
    this->output.append(buf);
  }

  this->return_code = WEXITSTATUS(pclose(this->fp));
}

int SubProcess::get_return_code() const { return this->return_code; }

const std::string& SubProcess::get_output() const { return this->output; }
}  // namespace profiler
}  // namespace tuning
}  // namespace mononn_engine