#include "mononn_engine/tuning/profiler/subprocess.h"

#include <cstdlib>

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
    LOG(FATAL) << "Subprocess: command " << command << " execution failed";
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