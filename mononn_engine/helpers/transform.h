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
#include <functional>
#include <vector>

namespace mononn_engine {
namespace helpers {
class Transform {
 public:
  //        template<typename Tin, typename Tout>
  //        static std::vector<Tout> map(std::vector<Tin> const &in,
  //        std::function<Tout(Tin const &element)> fn) {
  //            std::vector<Tout> result;
  //
  //            for (auto const &e : in) {
  //                result.push_back(fn(e));
  //            }
  //
  //            return result;
  //        }

  template <typename Tin, typename Tout>
  static std::vector<Tout> map(std::vector<Tin>& in,
                               std::function<Tout(Tin& element)> fn) {
    std::vector<Tout> result;

    for (auto& e : in) {
      result.push_back(fn(e));
    }

    return result;
  }

 private:
};
}  // namespace helpers
}  // namespace mononn_engine
