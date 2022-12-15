#pragma once

#include <string>

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
    struct Conv2dProblemSize {
        std::string N, H, W, C;
        std::string K, R, S;
        std::string pad_h, pad_w;
        std::string stride_h, stride_w;
        std::string dilation_h, dilation_w;
        std::string P, Q;
        std::string mode = "cutlass::conv::Mode::kCrossCorrelation";
        std::string split_k_slices = "1";
        std::string groups = "1";

        std::string define_variable(const std::string &var_name) const;
    };
}
}
}
}

