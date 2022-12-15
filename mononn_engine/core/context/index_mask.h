#pragma once
#include <string>
#include <vector>

namespace mononn_engine {
namespace core {
namespace context {
    class IndexMask {
    public:
        void add_mask(std::string _start, std::string _end);
    private:
        std::vector<std::string> start_index;
        std::vector<std::string> end_index;
    };
}
}
}


