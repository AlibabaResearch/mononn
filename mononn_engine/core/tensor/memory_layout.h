#pragma once 

#include <utility>
#include <vector>

namespace mononn_engine {
namespace core {
namespace tensor {
    class MemoryLayout {
    public:
        MemoryLayout() = default;
        MemoryLayout(std::vector<int> _perm) : perm(std::move(_perm)) {}
        bool valid() const;
        int rank() const;

        std::vector<int> get() const;
        int get(int index) const;

        std::vector<int> get_layout() const;
        int get_layout(int index) const;
    
        MemoryLayout flatten() const;
        MemoryLayout concat(const MemoryLayout &rhs) const;
        MemoryLayout reduce_dim(int index) const;
        MemoryLayout slice_dim(int start, int end) const;

        MemoryLayout permute(std::vector<int> _perm) const;

        std::string to_string() const;

        MemoryLayout normalize() const;

        bool operator == (const MemoryLayout& rhs) const;

    private:
        // Memory layout from first rank to last rank, 0 means the rank with fastest address variation. 
        // This is not following XLA HLO convention.
        std::vector<int> perm;

        void assert_layout_valid() const;
    }; 
}
}
}