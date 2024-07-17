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

#include "mononn_engine/core/context/index_transform.h"

#include "mononn_engine/helpers/string_helpers.h"

namespace mononn_engine {
namespace core {
namespace context {
std::vector<std::string> IndexTransform::offset_to_multi_index(
    TensorShape tensor_shape, std::string offset) {
  std::vector<std::string> multi_index;
  multi_index.resize(tensor_shape.rank());

  for (int idx = 0; idx < (int)multi_index.size(); ++idx) {
    int mod = tensor_shape.slice_dim(idx, -1).element_count();
    int div = tensor_shape.slice_dim(idx + 1, -1).element_count();

    if (tensor_shape.get_shape(idx) == 0) {
      multi_index[idx] = "0 /*index simplification*/";
      continue;
    }

    if (idx == 0) {
      multi_index[idx] = mononn_engine::helpers::string_format(
          "((%s) / (%d))", offset.c_str(), div);
    } else if (idx == (int)multi_index.size() - 1) {
      multi_index[idx] = mononn_engine::helpers::string_format(
          "((%s) %% (%d))", offset.c_str(), mod);
    } else {
      multi_index[idx] = mononn_engine::helpers::string_format(
          "(((%s) %% (%d)) / (%d))", offset.c_str(), mod, div);
    }
  }

  return multi_index;
}

std::string IndexTransform::multi_index_to_offset(
    TensorShape tensor_shape, std::vector<std::string> multi_index) {
  if (multi_index.empty()) {
    LOG(FATAL) << "Empty multi index.";
  }

  std::string offset;
  offset = multi_index[0];
  for (int idx = 1; idx < (int)multi_index.size(); ++idx) {
    offset = mononn_engine::helpers::string_format(
        "((%s) * (%s) + (%s))", offset.c_str(),
        std::to_string(tensor_shape.get_shape(idx)).c_str(),
        multi_index[idx].c_str());
  }

  return offset;
}
}  // namespace context
}  // namespace core
}  // namespace mononn_engine