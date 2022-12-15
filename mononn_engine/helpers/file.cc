#include <experimental/filesystem>
#include <fstream>
#include "mononn_engine/helpers/file.h"
#include "mononn_engine/helpers/macros.h"

namespace mononn_engine {
namespace helpers {
    namespace fs = std::experimental::filesystem;

    void File::copy(const std::string &src, const std::string &dst) {
        fs::copy_file(src, dst);
    }

    bool File::exists(const std::string &path) {
        return fs::exists(path);
    }

    void File::remove(const std::string &path) {
        fs::remove(path);
    }

    std::vector<uint8_t> File::read_as_binary(const std::string &path) {
        std::ifstream input(path, std::ios::binary);

        std::vector<uint8_t> buffer(std::istreambuf_iterator<char>(input), {});

        return std::move(buffer);
    }

    std::string File::read_as_string(const std::string &path) {
        std::ifstream ifs(path);
        std::stringstream buffer;
        buffer << ifs.rdbuf();

        return std::move(buffer.str());
    }

    void File::write_to_file(const std::string &content, const std::string &file) {
        std::ofstream ofs(file);

        if (!ofs.is_open()) {
            LOG(FATAL) << "Cannot open file " <<file;
        }

        ofs << content;
        ofs.close();
    }
}
}
