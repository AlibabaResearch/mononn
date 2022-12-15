#pragma once

#include <vector>
#include <string>

namespace mononn_engine {
namespace core {
namespace semantic {
    class FunctionInvocation {
    public:
        FunctionInvocation() {}
        FunctionInvocation(std::string _func_name) : func_name(_func_name) {}
        void add_template_arg(std::string template_arg);
        void add_arg(std::string arg);

        FunctionInvocation get_ilp_function_invocation(int ilp_id) const;

        std::string template_args_to_string() const;
        std::string args_to_string() const;
        std::string get_func_name() const;
        void set_func_name(std::string _func_name);
        void set_arg(int arg_id, std::string arg_name);
        std::string to_string() const;
    private:
        std::string func_name;
        std::vector<std::string> template_args;
        std::vector<std::string> args;
    };
}
}
}