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
namespace semantic {
class Using {
 public:
  Using(std::string _name, std::string _class_name)
      : name(_name), class_name(_class_name) {}
  Using(std::string _name, const char* _class_name)
      : Using(_name, std::string(_class_name)) {}
  Using(const char* _name, std::string _class_name)
      : Using(std::string(_name), _class_name) {}
  Using(const char* _name, const char* _class_name)
      : Using(std::string(_name), std::string(_class_name)) {}

  std::string get_name() const;
  void add_template_arg(std::string arg);
  std::string to_string() const;

  bool is_typename() const;
  void with(std::string _type);

 private:
  std::string name;
  std::string class_name;
  std::vector<std::string> template_args;
  std::vector<std::string> with_type;
};
}  // namespace semantic
}  // namespace core
}  // namespace mononn_engine