#pragma once

#include <vector>
#include <string>

namespace mononn_engine {
namespace tuning {
namespace profiler {
    class SubProcess {
    public:
        SubProcess() {};
        SubProcess(std::string const &_cmd) : cmd(_cmd) {}
        SubProcess(std::string const &_cmd, std::vector<std::string> const &_args) : cmd(_cmd), args(_args) {}

        void start();
        void wait();

        int get_return_code() const;
        const std::string &get_output() const;

    private:
        std::string cmd;
        std::vector<std::string> args;
        FILE *fp;
        int return_code;
        std::string output;
    };
}
}
}
