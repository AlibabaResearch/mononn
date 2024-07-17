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

#include "mononn_engine/tuning/profiler/profiling_result.h"

#include "mononn_engine/helpers/macros.h"

namespace mononn_engine {
namespace tuning {
namespace profiler {
void ProfilingResult::verify() const {
  if (!this->codegen_success) {
    LOG(FATAL) << "Codegen failed "
               << "\n"
               << this->output << "\n"
               << "Debug directory:" << this->profiling_directory;
  }

  if (!this->build_success) {
    LOG(FATAL) << "Build failed"
               << "\n"
               << this->output << "\n"
               << "Debug directory:" << this->profiling_directory;
  }

  if (!this->profile_success) {
    LOG(FATAL) << "Profile failed"
               << "\n"
               << this->output << "\n"
               << "Debug directory:" << this->profiling_directory;
  }
}
}  // namespace profiler
}  // namespace tuning
}  // namespace mononn_engine