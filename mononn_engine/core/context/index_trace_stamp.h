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
#include <map>
#include <string>
#include <vector>

namespace mononn_engine {
namespace core {
namespace context {
struct ConcreteIndexStamp {
  std::string index_before_trace;
  std::string index_after_trace;
  std::string traced_by;
  std::string pred_before_trace = "true";
  std::string pred_after_trace = "true";
  std::string value_on_false_pred = "0";

  bool operator==(ConcreteIndexStamp const& rhs) const;

  ConcreteIndexStamp instantiate(
      const std::map<std::string, std::string>& params) const;

  friend std::ostream& operator<<(std::ostream& os,
                                  ConcreteIndexStamp const& its);
};

struct SymbolicIndexStamp {
  std::string index_before_trace;
  std::string index_after_trace;
  std::string traced_by;
  std::string pred_before_trace = "true";
  std::string pred_after_trace = "true";
  std::string value_on_false_pred = "0";

  static SymbolicIndexStamp create(const std::string& _index_before_trace,
                                   const std::string& _index_after_trace,
                                   const std::string& _traced_by);

  ConcreteIndexStamp instantiate(
      const std::map<std::string, std::string>& params) const;

  bool operator==(SymbolicIndexStamp const& rhs) const;

  friend std::ostream& operator<<(std::ostream& os,
                                  SymbolicIndexStamp const& its);
};
}  // namespace context
}  // namespace core
}  // namespace mononn_engine
