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

#include "mononn_engine/helpers/path.h"

namespace mononn_engine {
namespace helpers {
class Directory {
 public:
  static std::string get_mononn_root_temp_dir();
  static std::string get_mononn_new_temp_dir();

  static bool exists(std::string dir);
  static void create(std::string dir);
  static void create_recursive(std::string dir);
  static void create_if_not_exists(std::string dir);
  static void create_recursive_if_not_exists(std::string dir);
  static void remove(std::string dir);
};

class TempDirectoryRAII {
 public:
  explicit TempDirectoryRAII(const std::string& _dir_name);

  const std::string& get_dir_name() const;

  ~TempDirectoryRAII();

 private:
  std::string dir_name;
};
}  // namespace helpers
}  // namespace mononn_engine
