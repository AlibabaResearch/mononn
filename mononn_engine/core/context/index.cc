#include "mononn_engine/core/context/index.h"

namespace mononn_engine {
namespace core {
namespace context {
    void Index::set(const std::string &_index) {
        this->index = _index;
    }

    const std::string &Index::get() const {
        return this->index;
    }

    const char *Index::c_str() const {
        return this->index.c_str();
    }

    bool Index::is_streaming_access() const {
        return this->index.find("%") == std::string::npos;
    }

    bool Index::is_streaming_access(const std::string &index) {
        return index.find("%") == std::string::npos;
    }
}
}
}