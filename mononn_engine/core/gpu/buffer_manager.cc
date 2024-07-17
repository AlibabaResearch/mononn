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

#include "mononn_engine/core/gpu/buffer_manager.h"

#include "mononn_engine/helpers/macros.h"

namespace mononn_engine {
namespace core {
namespace gpu {
// std::vector<std::string> BufferManager::variables_in_smem =
// std::vector<std::string>();
std::vector<std::string> thread_local BufferManager::variables_in_global =
    std::vector<std::string>();

std::string BufferManager::get_buffer_name(std::string var_name) {
  if (BufferManager::buffer_manager_use_tf_xla_buffer) {
    return var_name;
  }

  if (BufferManager::is_var_in_global(var_name))
    return BufferManager::get_buffer_name_global(var_name);

  if (var_name.rfind("get_tuple_element") == 0) {
    return var_name;
  }

  LOG(FATAL) << "Var " + var_name + " not in any buffer";
  return var_name;
}

std::string BufferManager::get_buffer_name_global(std::string var_name) {
  if (BufferManager::buffer_manager_use_tf_xla_buffer) {
    return var_name;
  }

  return var_name + "_buffer_global";
}

bool BufferManager::is_var_in_global(std::string var_name) {
  return std::find(BufferManager::variables_in_global.begin(),
                   BufferManager::variables_in_global.end(),
                   var_name) != BufferManager::variables_in_global.end();
}

void BufferManager::set_buffer_mnager_use_tf_xla_buffer(bool value) {
  BufferManager::buffer_manager_use_tf_xla_buffer = value;
}

void BufferManager::buffer_in_global(std::string var_name) {
  EXPECT_TRUE(!BufferManager::is_var_in_global(var_name),
              "Var " + var_name + " already exists in global buffer");
  BufferManager::variables_in_global.push_back(var_name);
}

void BufferManager::unbuffer_in_global(std::string var_name) {
  EXPECT_TRUE(BufferManager::is_var_in_global(var_name),
              "Var " + var_name + "not exists in global buffer");
  auto iter = std::find(BufferManager::variables_in_global.begin(),
                        BufferManager::variables_in_global.end(), var_name);
  BufferManager::variables_in_global.erase(iter);
}

std::vector<std::string> BufferManager::get_buffered_nodes_in_global() {
  return BufferManager::variables_in_global;
}

void BufferManager::reset() {
  // variables_in_smem.clear();
  variables_in_global.clear();
}

bool thread_local BufferManager::buffer_manager_use_tf_xla_buffer = false;
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine