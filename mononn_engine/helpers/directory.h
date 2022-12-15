#pragma once
#include <string>

#include "mononn_engine/helpers/path.h"

namespace mononn_engine {
namespace helpers {
    class Directory {
    public:
        static std::string get_mononn_root_temp_dir();
        static std::string get_mononn_new_temp_dir();

        static bool exists(std::string dir);
        static void create(std::string dir);
        static void create_recursive(std::string dir);
        static void create_if_not_exists(std::string dir);
        static void create_recursive_if_not_exists(std::string dir);
        static void remove(std::string dir);
    };

    class TempDirectoryRAII {
    public:
        explicit TempDirectoryRAII(const std::string &_dir_name);

        const std::string& get_dir_name() const;

        ~TempDirectoryRAII();
    private:
        std::string dir_name;
    };
}
}

