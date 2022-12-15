#pragma once

#include <vector>

namespace mononn_engine {
namespace core {
namespace common {
    class ILPNodeInterface {
    public:

        virtual void set_instruction_parallel_factor(int _ilp_factor) = 0;
        int get_instruction_parallel_factor() const;
//        virtual void set_ilp_traced_index(int ilp_id, std::vector<IndexTraceStamp> _traced_index_list);
//        virtual std::vector<IndexTraceStamp> get_ilp_traced_index(int ilp_id) const;
        bool is_instruction_parallelized() const;
//        bool is_node_ilp_traced(int ilp_id) const;

    protected:
//        std::vector<std::vector<SymbolicIndexStamp>> ilp_traced_index_list;
        int ilp_factor = 1;
    };
}
}
}


