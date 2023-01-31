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
namespace core {
namespace op_annotation {
class ClusterType {
 public:
  ClusterType() : name("None") {}
  ClusterType(std::string _name) : name(_name) {}
  ClusterType(const char* _name) : ClusterType(std::string(_name)) {}

  static const ClusterType None;
  static const ClusterType Reduce;
  static const ClusterType Elewise;
  static const ClusterType Gemm;
  static const ClusterType Conv;
  static const ClusterType GemmEpilogue;
  static const ClusterType ConvEpilogue;

  std::string to_string() const;

  bool operator==(ClusterType const& rhs) const;
  bool operator!=(ClusterType const& rhs) const;

 private:
  std::string name;
};
}  // namespace op_annotation
}  // namespace core
}  // namespace mononn_engine