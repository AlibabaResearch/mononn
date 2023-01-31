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

#include "mononn_engine/helpers/env_variable.h"

namespace mononn_engine {
namespace helpers {
bool EnvVar::defined(const std::string& env) {
  return getenv(env.c_str()) != nullptr;
}

bool EnvVar::is_true(const std::string& env) {
  if (!EnvVar::defined(env)) return false;

  std::string env_val = EnvVar::get(env);

  return env_val == "True" || env_val == "true" || env_val == "1";
}

std::string EnvVar::get(const std::string& env) { return getenv(env.c_str()); }

std::string EnvVar::get_with_default(const std::string& env,
                                     const std::string& default_value) {
  if (!EnvVar::defined(env)) return default_value;

  return EnvVar::get(env);
}
}  // namespace helpers
}  // namespace mononn_engine