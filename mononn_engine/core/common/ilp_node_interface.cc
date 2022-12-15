#include "mononn_engine/core/common/ilp_node_interface.h"
#include "mononn_engine/helpers/helpers.h"

namespace mononn_engine {
namespace core {
namespace common {

    int ILPNodeInterface::get_instruction_parallel_factor() const {
        return this->ilp_factor;
    }

//    void ILPNodeInterface::set_ilp_traced_index(int ilp_id, std::vector<IndexTraceStamp> _traced_index_list) {
//        if (ilp_id > this->ilp_traced_index_list.size()) {
//            LOG(FATAL) << "ILP id " << ilp_id << " out of range. Limit " << this->ilp_traced_index_list.size();
//        }
//
//        this->ilp_traced_index_list[ilp_id] = _traced_index_list;
//    }

//    std::vector<IndexTraceStamp> ILPNodeInterface::get_ilp_traced_index(int ilp_id) const {
//        if (ilp_id > this->ilp_traced_index_list.size()) {
//            LOG(FATAL) << "ILP id " << ilp_id << " out of range. Limit " << this->ilp_traced_index_list.size();
//        }
//
//        return this->ilp_traced_index_list[ilp_id];
//    }

    bool ILPNodeInterface::is_instruction_parallelized() const {
        return this->ilp_factor != 1;
    }

//    bool ILPNodeInterface::is_node_ilp_traced(int ilp_id) const {
//        if (this->ilp_traced_index_list.size() == 0) return false;
//
//        if (this->ilp_traced_index_list[ilp_id].size() == 0) return false;
//
//        return true;
//    }
}
}
}