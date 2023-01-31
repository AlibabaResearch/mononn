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

#include <cstdarg>
#include <string>

namespace mononn_engine {
namespace helpers {
class Path {
 public:
  static std::string join(std::string path1, std::string path2) {
    if (path1.back() == '/') path1 = path1.substr(0, path1.length() - 1);
    if (path2[0] == '/') path2 = path2.substr(1, path2.length() - 1);

    return path1 + "/" + path2;
  }

  template <typename... Args>
  static std::string join(std::string path1, std::string path2, Args... args) {
    return Path::join(Path::join(std::string(path1), std::string(path2)),
                      args...);
  }
};
}  // namespace helpers
}  // namespace mononn_engine