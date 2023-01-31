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

#include "mononn_engine/helpers/string_helpers.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <memory>

namespace mononn_engine {
namespace helpers {

std::string get_canonicalized_node_name(const std::string& name) {
  std::string var_name = name;

  for (int idx = 0; idx < (int)var_name.length(); ++idx) {
    if (var_name[idx] == '.' || var_name[idx] == '-') {
      var_name[idx] = '_';
    }
  }

  return var_name;
}

std::string get_hlo_module_short_name(const std::string& name) {
  if (name.substr(0, 8) != "cluster_") {
    LOG(FATAL) << "Invalid hlo module name " << name;
  }

  int pos = 8;
  while (std::isdigit(name[pos])) {
    ++pos;
    if (pos >= name.length()) {
      LOG(FATAL) << "Invalid hlo module name " << name;
    }
  }

  return name.substr(0, pos);
}

std::string string_to_lower(std::string str) {
  std::string ret = "";

  for (unsigned char c : str) {
    ret += std::tolower(c);
  }

  return ret;
}

std::string string_named_format(
    const std::string& format,
    const std::map<std::string, std::string>& names_and_values) {
  std::string result;
  std::string buf;
  bool record_name = false;

  for (const char& ch : format) {
    if (ch == '{') {
      record_name = true;
      continue;
    }

    if (ch == '}') {
      record_name = false;

      // auto name_and_value = std::find_if(names_and_values.begin(),
      // names_and_values.end(), [&](const std::pair<std::string, std::string>
      // &t) -> bool {
      //     return t.first == buf;
      // });

      // A symbol that not yet presented.
      if (!names_and_values.count(buf)) {
        // std::string debug_str;
        // for (auto const &[name, value] : names_and_values) debug_str +=
        // string_format("{%s, %s} ", name.c_str(), value.c_str()); LOG(FATAL)
        // << "Symbol " << buf << " cannot be found. Mappings: " << debug_str;

        // Leave it unchanged.
        result += "{" + buf + "}";
        continue;
      }

      result += names_and_values.at(buf);

      buf = "";

      continue;
    }

    if (record_name) {
      buf += ch;
    } else {
      result += ch;
    }
  }

  return result;
}

template <typename T>
std::string to_string(std::vector<T> const& vec) {
  std::stringstream ss;
  ss << "[";

  for (int idx = 0; idx < (int)vec.size(); ++idx) {
    if (idx == 0)
      ss << vec[idx];
    else {
      ss << ",";
      ss << vec[idx];
    }
  }

  ss << "]";

  return ss.str();
}

template std::string to_string<int32_t>(std::vector<int32_t> const& vec);
template std::string to_string<float>(std::vector<float> const& vec);
template std::string to_string<std::string>(
    std::vector<std::string> const& vec);
template std::string to_string<int64_t>(std::vector<int64_t> const& vec);

template <typename T>
std::string join(std::string sep, std::vector<T> const& vec) {
  std::string result;

  for (int idx = 0; idx < (int)vec.size(); ++idx) {
    result += std::to_string(vec[idx]);
    if (idx != (int)vec.size() - 1) result += sep;
  }

  return result;
}

template <>
std::string join<std::string>(std::string sep,
                              std::vector<std::string> const& vec) {
  std::string result;

  for (int idx = 0; idx < (int)vec.size(); ++idx) {
    result += vec[idx];
    if (idx != (int)vec.size() - 1) result += sep;
  }

  return result;
}

std::vector<std::string> string_split(const std::string& str, char sep) {
  std::vector<std::string> result;

  int last_idx = 0;
  for (int idx = 0; idx < str.length(); ++idx) {
    if (str[idx] == sep) {
      if (last_idx != idx) {
        result.push_back(str.substr(last_idx, idx - last_idx));
      }

      last_idx = idx + 1;
    }
  }

  if (last_idx != str.length()) {
    result.push_back(str.substr(last_idx, str.length() - last_idx));
  }

  return result;
}

std::string get_node_ilp_name(std::string node_name, int ilp_id) {
  return node_name + "__i" + std::to_string(ilp_id);
}

template std::string join<int32_t>(std::string sep,
                                   std::vector<int> const& vec);
template std::string join<float>(std::string sep,
                                 std::vector<float> const& vec);
template std::string join<int64_t>(std::string sep,
                                   std::vector<int64_t> const& vec);
}  // namespace helpers
}  // namespace mononn_engine