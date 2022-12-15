#include "mononn_engine/core/computation/computation.h"

namespace mononn_engine {
namespace core {
namespace computation {
    Computation::Computation(xla::HloComputation *computation) {
        this->name = computation->name();
    }


    std::string Computation::get_name() const {
        return this->name;
    }
}
}
}