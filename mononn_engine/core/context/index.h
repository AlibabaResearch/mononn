#pragma once
#include <string>

namespace mononn_engine {
namespace core {
namespace context {
    class Index {
    public:
        Index() {};
        Index(const std::string &_index) : index(_index) {}

        void set(const std::string &_index);
        const std::string &get() const;
        const char *c_str() const;

        bool is_streaming_access() const;

        static bool is_streaming_access(const std::string &index);
    private:
        std::string index;
    };
}
}
}
