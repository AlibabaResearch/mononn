
#include "mononn_engine/core/common/ilp_node_impl_interface.h"

namespace mononn_engine {
namespace core {
namespace common {
    using ConcreteIndexStamp = mononn_engine::core::context::ConcreteIndexStamp;
    using SymbolicIndexStamp = mononn_engine::core::context::SymbolicIndexStamp;

    std::vector<ConcreteIndexStamp> ILPNodeImplInterface::get_ilp_concrete_index(int ilp_id) {
        return this->ilp_concrete_index_list[ilp_id];
    }

    ConcreteIndexStamp ILPNodeImplInterface::get_ilp_concrete_index(int ilp_id, int index_id) {
        return this->ilp_concrete_index_list[ilp_id][index_id];
    }
}
}
}