#pragma once 

#include <string>

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
    class Layout {
    public:
        Layout() {}
        Layout(std::string _name) : name(_name) {}
        Layout(const char * _name) : Layout(std::string(_name)) {}

        std::string to_string() const;

        static Layout const RowMajor;
        static Layout const ColumnMajor;
        static Layout const TensorNHWC;
        static Layout const TensorNCHW;

        bool operator == (Layout const& rhs) const;
        bool operator != (Layout const& rhs) const;
    private:
        std::string name;
    };
}
}
}
}