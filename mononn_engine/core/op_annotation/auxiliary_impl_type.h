#pragma once
#include <string>

namespace mononn_engine {
namespace core {
namespace op_annotation {
    class AuxiliaryImplType {
    public:
        static const std::string buffer_in_register;
        static const std::string explicit_output_node;
        static const std::string cache_prefetch;
    private:

    };
}
}
}