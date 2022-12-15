#pragma once

namespace mononn_engine {
namespace core {
namespace common {
    struct CompileOutputType {
    public:
        enum Type {
            COMPILE_OUTPUT_TYPE_PTX,
            COMPILE_OUTPUT_TYPE_CUBIN,
            COMPILE_OUTPUT_TYPE_BINARY,
        };

    private:

    };
}
}
}