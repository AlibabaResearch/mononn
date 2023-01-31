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

#include "mononn_engine/helpers/directory.h"

#include <chrono>
#include <experimental/filesystem>

#include "mononn_engine/helpers/macros.h"
#include "mononn_engine/helpers/path.h"
#include "mononn_engine/helpers/uuid.h"

namespace fs = std::experimental::filesystem;

namespace mononn_engine {
namespace helpers {
std::string Directory::get_mononn_root_temp_dir() {
  return "/tmp/mononn_codegen";
}

std::string Directory::get_mononn_new_temp_dir() {
  return mononn_engine::helpers::Path::join(
      Directory::get_mononn_root_temp_dir(),
      mononn_engine::helpers::UUID::new_uuid());
}

bool Directory::exists(std::string dir) {
  return fs::exists(dir) && fs::is_directory(dir);
}

void Directory::create(std::string dir) {
  if (Directory::exists(dir)) {
    LOG(DEBUG) << "Directory " << dir << " already exists.";
  }

  fs::create_directory(dir);
}

void Directory::create_recursive(std::string dir) {
  if (Directory::exists(dir)) {
    LOG(DEBUG) << "Directory " << dir << " already exists.";
  }

  fs::create_directories(dir);
}

void Directory::create_if_not_exists(std::string dir) {
  if (!Directory::exists(dir)) {
    Directory::create(dir);
  }
}

void Directory::create_recursive_if_not_exists(std::string dir) {
  if (!Directory::exists(dir)) {
    Directory::create_recursive(dir);
  }
}

void Directory::remove(std::string dir) {
  if (!Directory::exists(dir)) {
    LOG(FATAL) << "Directory " << dir << " not exists";
  }

  fs::remove_all(dir);
}

TempDirectoryRAII::TempDirectoryRAII(const std::string& _dir_name)
    : dir_name(_dir_name) {
  Directory::create_recursive_if_not_exists(this->dir_name);
}

const std::string& TempDirectoryRAII::get_dir_name() const {
  return this->dir_name;
}

TempDirectoryRAII::~TempDirectoryRAII() { Directory::remove(this->dir_name); }
}  // namespace helpers
}  // namespace mononn_engine