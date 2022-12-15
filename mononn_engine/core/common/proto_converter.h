#pragma once

#include <memory>

namespace mononn_engine {
namespace core {
namespace common {
    template<typename T>
    class ProtoConverter {
    public:
        virtual std::unique_ptr<T> ConvertToProto() const = 0;
        virtual void ParseFromProto(T const *) = 0;
    };
}
}
}

