#pragma once

#include <string>

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {

    class StrideSupport {
    public:
        StrideSupport(const std::string &_name) : name(_name) {}
        StrideSupport(const char *_name) : name(std::string(_name)) {}

        static const StrideSupport kStrided;
        static const StrideSupport kUnity;

        std::string to_string() const;
    private:
        std::string name;
    };
}
}
}
}
