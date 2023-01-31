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
namespace core {
namespace gpu {
class BufferManager {
 public:
  BufferManager() {}

  static std::string get_buffer_name(std::string var_name);

  static bool is_var_in_global(std::string var_name);

  static void buffer_in_global(std::string var_name);

  static std::vector<std::string> get_buffered_nodes_in_global();

  static void reset();

  static void set_buffer_mnager_use_tf_xla_buffer(bool value);

 private:
  // static std::vector<std::string> variables_in_smem;

  static void unbuffer_in_global(std::string var_name);
  static std::string get_buffer_name_global(std::string var_name);

  // Buffer manager may used by multiple threads, use thread local as a interim
  // solution.
  static thread_local std::vector<std::string> variables_in_global;
  static thread_local bool buffer_manager_use_tf_xla_buffer;
};
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine