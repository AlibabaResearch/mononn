#pragma once
#include <string>
#include <vector>

namespace mononn_engine {
namespace parser {
    class HloModuleDumper {
    public:
        HloModuleDumper(
                const std::string &_frozen_pb_file,
                const bool &_auto_mixed_precision,
                const std::vector<std::string> &_feeds,
                const std::vector<std::string> &_input_files,
                const std::vector<std::string> &_fetches) :
                frozen_pb_file(_frozen_pb_file), auto_mixed_precision(_auto_mixed_precision), feeds(_feeds), input_files(_input_files), fetches(_fetches) {}

        void dump(std::string binary_path, std::string text_path = "");

    private:
        std::string frozen_pb_file;
        bool auto_mixed_precision;
        std::vector<std::string> feeds;
        std::vector<std::string> input_files;
        std::vector<std::string> fetches;
    };
}
}