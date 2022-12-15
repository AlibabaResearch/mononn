#include "mononn_engine/core/op/op_type.h"

namespace mononn_engine {
namespace core {
namespace op {
    std::string OpType::get_name() const {
        return this->name;
    }

    std::string OpType::to_string() const {
        return this->name;
    }

    #define DEFINE_OP_TYPE(op_name, op_code, ...) \
    const OpType OpType::op_code = OpType(op_name);
    OP_TYPE_LIST(DEFINE_OP_TYPE)
    OP_TYPE_LIST_CLUSTER(DEFINE_OP_TYPE)
    OP_TYPE_LIST_ONE_FUSER_ADDON(DEFINE_OP_TYPE)

    #undef DEFINE_OP_TYPE

    bool OpType::operator== (const OpType &rhs) const {
        return this->name == rhs.name;
    }

    bool OpType::operator!= (const OpType &rhs) const {
        return this->name != rhs.name;
    }

    bool OpType::operator< (const OpType &rhs) const {
        return this->name < rhs.name;
    }
}
}
}