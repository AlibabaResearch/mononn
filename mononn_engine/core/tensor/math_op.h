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
#include <unordered_map>

namespace mononn_engine {
namespace core {
namespace tensor {
class MathOp {
 public:
  MathOp() {}
  MathOp(std::string _op) : op(_op) {}
  MathOp(const char* _op) : MathOp(std::string(_op)) {}
  std::string to_string() const;

  static const MathOp assign;
  static const MathOp plus_assign;
  static const MathOp equal_to;
  static const MathOp not_equal_to;
  static const MathOp greater_equal_than;
  static const MathOp greater_than;
  static const MathOp less_equal_than;
  static const MathOp less_than;

  static MathOp from_string(std::string str);
  static std::unordered_map<std::string, MathOp>* registry();

  bool operator==(MathOp const& rhs) const;

 private:
  std::string op;
};
}  // namespace tensor
}  // namespace core
}  // namespace mononn_engine