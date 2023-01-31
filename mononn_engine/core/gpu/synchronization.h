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
namespace gpu {
class Synchronization {
 public:
  Synchronization() {}
  Synchronization(std::string _type, int _synchronization_level)
      : type(_type), synchronization_level(_synchronization_level) {}
  Synchronization(const char* _type, int _synchronization_level)
      : Synchronization(std::string(_type), _synchronization_level) {}

  static Synchronization const None;
  static Synchronization const Warp;
  static Synchronization const ThreadBlock;
  static Synchronization const Global;

  std::string to_string() const;

  bool operator<(const Synchronization& rhs) const;
  bool operator>(const Synchronization& rhs) const;
  bool operator!=(const Synchronization& rhs) const;
  bool operator==(const Synchronization& rhs) const;

  static std::string get_prerequisite_definition();

 private:
  std::string type;
  int synchronization_level;
};
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine