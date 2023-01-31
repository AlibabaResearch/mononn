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
class FunctionInvocation {
 public:
  FunctionInvocation() {}
  FunctionInvocation(std::string _func_name) : func_name(_func_name) {}
  void add_template_arg(std::string template_arg);
  void add_arg(std::string arg);

  FunctionInvocation get_ilp_function_invocation(int ilp_id) const;

  std::string template_args_to_string() const;
  std::string args_to_string() const;
  std::string get_func_name() const;
  void set_func_name(std::string _func_name);
  void set_arg(int arg_id, std::string arg_name);
  std::string to_string() const;

 private:
  std::string func_name;
  std::vector<std::string> template_args;
  std::vector<std::string> args;
};
}  // namespace semantic
}  // namespace core
}  // namespace mononn_engine