#pragma once
#include <map>
#include <memory>
#include <sstream>
#include <string>

#include "mononn_engine/helpers/macros.h"

namespace mononn_engine {
namespace helpers {

std::string get_canonicalized_node_name(const std::string& name);
std::string get_hlo_module_short_name(const std::string& name);

std::string string_to_lower(std::string str);

template <typename... Args>
std::string string_format(const std::string& format, Args... args) {
  int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1;
  size_t size = static_cast<size_t>(size_s);
  auto buf = std::make_unique<char[]>(size);
  std::snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(buf.get(), size - 1);
}

std::string string_named_format(
    const std::string& format,
    const std::map<std::string, std::string>& names_and_values);

template <typename T>
std::string to_string(const std::vector<T>& vec);

template <typename T>
std::string join(std::string sep, const std::vector<T>& vec);

template <typename T>
std::string join(std::string sep, const std::vector<T>& vec,
                 std::function<std::string(const T&)> string_converter) {
  std::vector<std::string> str_list;

  for (auto const& item : vec) {
    str_list.push_back(string_converter(item));
  }

  return join(sep, str_list);
}

std::vector<std::string> string_split(const std::string& str, char sep);

std::string get_node_ilp_name(std::string node_name, int ilp_id);
}  // namespace helpers
}  // namespace mononn_engine