#include "mononn_engine/core/gpu/smem_manager.h"

#include "mononn_engine/helpers/string_helpers.h"

namespace mononn_engine {
namespace core {
namespace gpu {
int round_up_to(size_t num, size_t factor) {
  return num + (factor - (num % factor)) % factor;
}

std::string SmemManager::define_root_buffer() const {
  return mononn_engine::helpers::string_format(
      "extern __shared__ int8_t %s[];\n", this->root_buffer_ptr.c_str());
}

bool SmemManager::can_claim_buffer(size_t size_in_bytes) const {
  size_t current_size = (size_t)this->multi_buffer.get_total_size_in_bytes();
  current_size = round_up_to(current_size, this->alignment_in_byte);

  return (current_size + size_in_bytes) <= this->total_size_in_bytes;
}

bool SmemManager::can_claim_buffer(
    const std::vector<size_t>& size_list_in_bytes) const {
  size_t current_size = (size_t)this->multi_buffer.get_total_size_in_bytes();

  for (auto size_in_bytes : size_list_in_bytes) {
    current_size = round_up_to(
        current_size, this->alignment_in_byte);  // Ordering is correct, dont
                                                 // need to round last buffer.
    current_size += size_in_bytes;
  }

  return current_size <= this->total_size_in_bytes;
}

void SmemManager::claim_smem_buffer(std::string buffer_name,
                                    size_t size_in_bytes) {
  if (this->buffer_id.count(buffer_name)) {
    LOG(FATAL) << "Buffer: " << buffer_name << " already claimed.";
  }

  this->buffer_id[buffer_name] = this->current_buffer_id;
  this->buffer_size[buffer_name] = size_in_bytes;

  this->multi_buffer.add_buffer(size_in_bytes);

  if (this->multi_buffer.get_total_size_in_bytes() >
      this->total_size_in_bytes) {
    LOG(FATAL) << "Current buffer size in bytes "
               << this->multi_buffer.get_total_size_in_bytes()
               << " exceeds buffer size limit " << this->total_size_in_bytes;
  }

  this->current_buffer_id += 1;
};

void SmemManager::claim_smem_buffer(std::string node_name,
                                    std::string buffer_name,
                                    size_t size_in_bytes) {
  this->claim_smem_buffer(buffer_name, size_in_bytes);

  this->node_name_to_smem_buffer_name[node_name] = buffer_name;
};

std::string SmemManager::get_buffer_pointer(std::string buffer_name,
                                            std::string as_type) const {
  if (!this->buffer_id.count(buffer_name)) {
    LOG(FATAL) << "Buffer: " << buffer_name << " not found.";
  }

  return this->multi_buffer.get_pointer_to_buffer(
      this->buffer_id.at(buffer_name), as_type);
}

std::string SmemManager::get_buffer_name(std::string node_name) const {
  return this->node_name_to_smem_buffer_name.at(node_name);
}

size_t SmemManager::get_buffer_size(const std::string& buffer_name) const {
  return this->buffer_size.at(buffer_name);
}

void SmemManager::record_base_buffer_size(const std::string& buffer_name,
                                          size_t size_in_bytes) {
  this->base_buffer_size[buffer_name] = size_in_bytes;
}

size_t SmemManager::get_base_buffer_size(const std::string& buffer_name) const {
  if (!this->base_buffer_size.count(buffer_name)) {
    LOG(FATAL) << "Key not exist: " << buffer_name;
  }

  return this->base_buffer_size.at(buffer_name);
}
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine