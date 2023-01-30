#pragma once

#include <string>
#include <unordered_map>

#include "mononn_engine/core/gpu/multi_buffer.h"

namespace mononn_engine {
namespace core {
namespace gpu {
class SmemManager {
 public:
  SmemManager(size_t _total_size_in_bytes)
      : SmemManager(_total_size_in_bytes, "__cache", 4) {}
  SmemManager(size_t _total_size_in_bytes, std::string _root_buffer_ptr,
              int _alignment_in_byte)
      : total_size_in_bytes(_total_size_in_bytes),
        alignment_in_byte(_alignment_in_byte),
        multi_buffer(_root_buffer_ptr, _alignment_in_byte),
        root_buffer_ptr(_root_buffer_ptr) {}

  std::string define_root_buffer() const;
  bool can_claim_buffer(size_t size_in_bytes) const;
  bool can_claim_buffer(const std::vector<size_t>& size_in_bytes) const;
  void claim_smem_buffer(std::string buffer_name, size_t size_in_bytes);
  void claim_smem_buffer(std::string node_name, std::string buffer_name,
                         size_t size_in_bytes);
  std::string get_buffer_pointer(std::string buffer_name,
                                 std::string as_type = "void *") const;
  std::string get_buffer_name(std::string node_name) const;

  size_t get_buffer_size(const std::string& buffer_name) const;

  // Base buffer size is the size of the highest dimension.
  void record_base_buffer_size(const std::string& buffer_name,
                               size_t size_in_bytes);
  size_t get_base_buffer_size(const std::string& buffer_name) const;

 private:
  MultiBuffer multi_buffer;
  std::string root_buffer_ptr;
  std::unordered_map<std::string, int> buffer_id;
  std::unordered_map<std::string, size_t> buffer_size;
  std::unordered_map<std::string, size_t> base_buffer_size;
  int current_buffer_id = 0;
  size_t total_size_in_bytes;
  int alignment_in_byte;
  std::unordered_map<std::string, std::string> node_name_to_smem_buffer_name;
};
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine