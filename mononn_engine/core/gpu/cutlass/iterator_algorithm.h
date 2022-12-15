#pragma once

#include <string>

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {

    class IteratorAlgorithm {
    public:
        IteratorAlgorithm(const std::string &_name) : name(_name) {}
        IteratorAlgorithm(const char *_name) : name(std::string(_name)) {}

        static const IteratorAlgorithm kOptimized;
        static const IteratorAlgorithm kAnalytic;

        std::string to_string() const;
    private:
        std::string name;
    };
}
}
}
}
