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

namespace mononn_engine {
namespace helpers {
class EnvVar {
 public:
  static bool defined(const std::string& env);
  static bool is_true(const std::string& env);
  static std::string get(const std::string& env);
  static std::string get_with_default(const std::string& env,
                                      const std::string& default_value = "");

 private:
};
}  // namespace helpers
}  // namespace mononn_engine