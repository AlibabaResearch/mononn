#pragma once

#include <string>
#include "mononn_engine/core/gpu/dim3.h"
#include "mononn_engine/core/semantic/function_invocation.h"

namespace mononn_engine {
namespace core {
namespace semantic {
    class CUDAInvocation {
    public:
        using Dim3 = mononn_engine::core::gpu::Dim3;
        CUDAInvocation(std::string _func_name, Dim3 _grid, Dim3 _block, int _smem_size, std::string _stream)
        : function_invocation(_func_name), grid(_grid), block(_block), smem_size(_smem_size), stream(_stream) {}

        void add_template_arg(std::string template_arg);
        void add_arg(std::string arg);

        std::string cuda_config_to_string() const;
        std::string to_string() const;

    private:
        FunctionInvocation function_invocation;
        Dim3 grid;
        Dim3 block;
        int smem_size;
        std::string stream;
    };
}
}
}