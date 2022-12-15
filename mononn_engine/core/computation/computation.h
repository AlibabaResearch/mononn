#pragma once

#include <string>

#include "tensorflow/compiler/xla/service/hlo_computation.h"

namespace mononn_engine {
namespace core {
namespace computation {
    class Computation {
    public:
        Computation() {};
        Computation(xla::HloComputation *computation);
        std::string get_name() const;
    private:
        std::string name;
    };
}
}
}