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