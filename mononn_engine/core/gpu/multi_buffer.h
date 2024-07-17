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

#include <vector>

#include "mononn_engine/core/tensor/tensor_spec.h"

namespace mononn_engine {
namespace core {
namespace gpu {
class MultiBuffer {
 public:
  using TensorSpec = mononn_engine::core::tensor::TensorSpec;
  MultiBuffer(std::string _buffer_ptr)
      : buffer_ptr(_buffer_ptr), alignment_in_byte(256) {}
  MultiBuffer(std::string _buffer_ptr, int _alignment_in_byte)
      : buffer_ptr(_buffer_ptr), alignment_in_byte(_alignment_in_byte) {}

  void add_buffer(TensorSpec tensor_spec);
  void add_buffer(int64_t buffer_size_in_bytes);

  int64_t get_total_size_in_bytes() const;

  std::string get_pointer_to_buffer(int buffer_id,
                                    std::string as_type = "void *") const;

 private:
  std::vector<int64_t> buffer_size_list;
  std::string buffer_ptr;
  int alignment_in_byte;
};
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine
