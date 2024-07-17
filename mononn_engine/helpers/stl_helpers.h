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
#include <memory>
#include <vector>

namespace mononn_engine {
namespace helpers {
template <typename T1, typename T2, typename TRes>
std::vector<TRes> cartesian_join(
    const std::vector<T1>& vec1, const std::vector<T2>& vec2,
    std::function<TRes(const T1&, const T2&)> func) {
  std::vector<TRes> res;

  for (auto const& elem1 : vec1) {
    for (auto const& elem2 : vec2) {
      res.push_back(func(elem1, elem2));
    }
  }

  return res;
}

template <typename T>
std::vector<T> vector_concat(std::vector<T> const& vec1,
                             std::vector<T> const& vec2) {
  std::vector<T> result = vec1;
  result.insert(result.end(), vec2.begin(), vec2.end());
  return result;
}

template <typename T, typename... TArgs>
std::vector<T> vector_concat(std::vector<T> const& vec1,
                             std::vector<T> const& vec2,
                             TArgs... additional_vec) {
  return vector_concat(vector_concat(vec1, vec2), additional_vec...);
}

template <typename Derived, typename Base, typename Del>
std::unique_ptr<Derived, Del> static_unique_ptr_cast(
    std::unique_ptr<Base, Del>&& p) {
  auto d = static_cast<Derived*>(p.release());
  return std::unique_ptr<Derived, Del>(d);
}
}  // namespace helpers
}  // namespace mononn_engine
