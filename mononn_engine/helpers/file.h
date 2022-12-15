#pragma once

#include <string>
#include <vector>

namespace mononn_engine {
namespace helpers {
    class File {
    public:
        static void copy(const std::string &src, const std::string &dst);
        static bool exists(const std::string &path);
        static void remove(const std::string &path);
        static std::vector<uint8_t> read_as_binary(const std::string &path);
        static std::string read_as_string(const std::string &path);

        static void write_to_file(const std::string &content, const std::string &file);
    };
}
}