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

#include "mononn_engine/core/gpu/multi_buffer.h"

#include "mononn_engine/helpers/string_helpers.h"

namespace mononn_engine {
namespace core {
namespace gpu {
int64_t div_up(int64_t num, int64_t div) { return (num + div - 1) / div; }

int64_t round_up_to(int64_t num, int64_t factor) {
  return div_up(num, factor) * factor;
}

void MultiBuffer::add_buffer(MultiBuffer::TensorSpec tensor_spec) {
  this->buffer_size_list.push_back(tensor_spec.size_in_bytes());
}

void MultiBuffer::add_buffer(int64_t buffer_size_in_bytes) {
  this->buffer_size_list.push_back(buffer_size_in_bytes);
}

int64_t MultiBuffer::get_total_size_in_bytes() const {
  int64_t total_size_in_bytes = 0;
  for (auto const& buffer_size : this->buffer_size_list) {
    total_size_in_bytes +=
        round_up_to(buffer_size, (int64_t)this->alignment_in_byte);
  }

  return total_size_in_bytes;
}

std::string MultiBuffer::get_pointer_to_buffer(int buffer_id,
                                               std::string as_type) const {
  int64_t offset_in_bytes = 0;
  for (int idx = 0; idx < buffer_id; ++idx) {
    offset_in_bytes += round_up_to(this->buffer_size_list[idx],
                                   (int64_t)this->alignment_in_byte);
  }

  std::string ptr = mononn_engine::helpers::string_format(
      "(&reinterpret_cast<int8_t *>(%s)[%s])", this->buffer_ptr.c_str(),
      std::to_string(offset_in_bytes).c_str());

  return mononn_engine::helpers::string_format("reinterpret_cast<%s>(%s)",
                                               as_type.c_str(), ptr.c_str());
}
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine