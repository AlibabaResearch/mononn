#pragma once

#include <string>

namespace mononn_engine {
namespace helpers {
    class EnvVar {
    public:
        static bool defined(const std::string &env);
        static bool is_true(const std::string &env);
        static std::string get(const std::string &env);
        static std::string get_with_default(const std::string &env, const std::string &default_value = "");
    private:
    }; 
}
}