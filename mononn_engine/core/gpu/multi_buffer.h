#pragma once

#include <vector>
#include "mononn_engine/core/tensor/tensor_spec.h"

namespace mononn_engine {
namespace core {
namespace gpu {
    class MultiBuffer {
    public:
        using TensorSpec = mononn_engine::core::tensor::TensorSpec;
        MultiBuffer(std::string _buffer_ptr) : buffer_ptr(_buffer_ptr), alignment_in_byte(256) {}
        MultiBuffer(std::string _buffer_ptr, int _alignment_in_byte)
            : buffer_ptr(_buffer_ptr), alignment_in_byte(_alignment_in_byte) {}

        void add_buffer(TensorSpec tensor_spec);
        void add_buffer(int64_t buffer_size_in_bytes);

        int64_t get_total_size_in_bytes() const;

        std::string get_pointer_to_buffer(int buffer_id, std::string as_type = "void *") const;
    private:
    
        std::vector<int64_t> buffer_size_list;
        std::string buffer_ptr;
        int alignment_in_byte;
    };
}
}
}
