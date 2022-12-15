#pragma once  
#include <string>
#include <memory>
#include <sstream>
#include <map>

#include "mononn_engine/core/context/cuda_context.h"
#include "mononn_engine/codegen/model_data.h"
#include "mononn_engine/core/common/compile_output_type.h"

namespace mononn_engine {
namespace codegen {
    // A self-contained runnable object including source code, compile, link, and build functionality.
    class CUDAProgram {
    public:
        using CUDAContext = mononn_engine::core::context::CUDAContext;
        using CompileOutputType = mononn_engine::core::common::CompileOutputType::Type;

        CUDAProgram() {}
        CUDAProgram(std::shared_ptr<CUDAContext> _cuda_context) : cuda_context(_cuda_context) {}

        void append_main(std::string const &content);
        void append_file(std::string const &file_name, std::string const &content);
        std::vector<std::string> get_file_list() const;
        std::stringstream& file_ref(std::string const &file_name);
        const std::stringstream& file_ref(std::string const &file_name) const;

        void generate(std::string const &directory, CompileOutputType type = CompileOutputType::COMPILE_OUTPUT_TYPE_BINARY) const;
        bool build(std::string const &directory) const;
        bool generate_and_build(std::string const &directory) const;

        void add_model_data(std::unique_ptr<ModelData> model_data);
        void add_data_file(std::string data_file);
    private:
        std::shared_ptr<CUDAContext> cuda_context;
        std::map<std::string, std::stringstream> files;
        std::vector<std::unique_ptr<ModelData>> model_data_list;
        std::string get_makefile(CompileOutputType type) const;
        // std::string get_cmakefile(bool build_cubin) const;
        std::vector<std::string> data_files;
    };
}
}