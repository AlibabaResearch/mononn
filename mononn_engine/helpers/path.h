#pragma once

#include <string>
#include <cstdarg>

namespace mononn_engine {
namespace helpers {
    class Path {
    public:
        static std::string join(std::string path1, std::string path2) {
            if (path1.back() == '/') path1 = path1.substr(0, path1.length() - 1);
            if (path2[0] == '/') path2 = path2.substr(1, path2.length() - 1);

            return path1 + "/" + path2;
        }

        template <typename ... Args>
        static std::string join(std::string path1, std::string path2, Args ... args) {
            return Path::join(Path::join(std::string(path1), std::string (path2)), args...);
        }
    };
}
}