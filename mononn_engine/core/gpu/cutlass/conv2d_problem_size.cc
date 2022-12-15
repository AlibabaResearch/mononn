#include <sstream>
#include "mononn_engine/core/gpu/cutlass/conv2d_problem_size.h"

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
    std::string Conv2dProblemSize::define_variable(const std::string &var_name) const {
        std::stringstream ss;
        ss << "cutlass::conv::Conv2dProblemSize " << var_name << "(" << "\n";
        ss << "{" << this->N << "," << this->H << "," << this->W << "," << this->C << "}," << "\n";
        ss << "{" << this->K << "," << this->R << "," << this->S << "," << this->C << "}," << "\n";
        ss << "{" << this->pad_h << "," << this->pad_h << "," << this->pad_w << "," << this->pad_w << "}," << "\n";
        ss << "{" << this->stride_h << "," << this->stride_w  << "}," << "\n";
        ss << "{" << this->dilation_h << "," << this->dilation_w  << "}," << "\n";
        ss << "{" << this->N << "," << this->P << "," << this->Q << "," << this->K << "}," << "\n";
        ss << "cutlass::conv::Mode::kCrossCorrelation," << "\n";
        ss << "1" << "\n";
        ss << ");" << "\n";
        return ss.str();
    }
}
}
}
}