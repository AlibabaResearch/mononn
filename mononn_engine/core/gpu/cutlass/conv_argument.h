#pragma once

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
    struct ConvArgument {
        std::string problem_size;
        std::string ptr_a;
        std::string ptr_b;
        std::string ptr_c;
        std::string ptr_d;

        std::string alpha, beta;

        std::string define_variable(const std::string &kernel_name, const std::string &var_name) const;
    };
}
}
}
}


