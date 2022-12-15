#pragma once 

#include <string>

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
    class GemmCoord {
    public:
        GemmCoord() {}
        GemmCoord(int _m, int _n, int _k) : m(_m), n(_n), k(_k) {}
        
        std::string to_string() const;
    private:
        int m, n, k;
    };
}
}
}
}