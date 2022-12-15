#include "mononn_engine/core/context/index_mask.h"

namespace mononn_engine {
namespace core {
namespace context {
    void IndexMask::add_mask(std::string _start, std::string _end) {
        this->start_index.push_back(_start);
        this->end_index.push_back(_end);
    }
}
}
}