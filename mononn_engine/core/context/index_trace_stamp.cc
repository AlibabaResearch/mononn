#include "mononn_engine/core/context/index_trace_stamp.h"
#include "mononn_engine/helpers/string_helpers.h"

namespace mononn_engine {
namespace core {
namespace context {
    bool ConcreteIndexStamp::operator==(const ConcreteIndexStamp &rhs) const  {
        return this->index_before_trace == rhs.index_before_trace &&
               this->index_after_trace == rhs.index_after_trace &&
               this->traced_by == rhs.traced_by;
    }

    ConcreteIndexStamp ConcreteIndexStamp::instantiate(const std::map<std::string, std::string> &params) const {
        ConcreteIndexStamp concrete_index_stamp;
        concrete_index_stamp.index_before_trace =
                mononn_engine::helpers::string_named_format(this->index_before_trace, params);
        concrete_index_stamp.index_after_trace =
                mononn_engine::helpers::string_named_format(this->index_after_trace, params);
        concrete_index_stamp.traced_by = this->traced_by;
        concrete_index_stamp.pred_before_trace =
                mononn_engine::helpers::string_named_format(this->pred_before_trace, params);
        concrete_index_stamp.pred_after_trace =
                mononn_engine::helpers::string_named_format(this->pred_after_trace, params);
        concrete_index_stamp.value_on_false_pred = this->value_on_false_pred;

        return concrete_index_stamp;
    }

    std::ostream& operator << (std::ostream &os, ConcreteIndexStamp const &its) {
        os << std::string("ConcreteIndexStamp: [Before trace: ")
           << its.index_before_trace
           << std::string(", after trace: ")
           << its.index_after_trace
           << std::string(", traced by: ")
           << its.traced_by
           << std::string("]");

        return os;
    }

    ConcreteIndexStamp SymbolicIndexStamp::instantiate(const std::map<std::string, std::string> &params) const {
        ConcreteIndexStamp concrete_index_stamp;
        concrete_index_stamp.index_before_trace =
                mononn_engine::helpers::string_named_format(this->index_before_trace, params);
        concrete_index_stamp.index_after_trace =
                mononn_engine::helpers::string_named_format(this->index_after_trace, params);
        concrete_index_stamp.traced_by = this->traced_by;
        concrete_index_stamp.pred_before_trace =
                mononn_engine::helpers::string_named_format(this->pred_before_trace, params);
        concrete_index_stamp.pred_after_trace =
                mononn_engine::helpers::string_named_format(this->pred_after_trace, params);
        concrete_index_stamp.value_on_false_pred = this->value_on_false_pred;

        return concrete_index_stamp;
    }

    bool SymbolicIndexStamp::operator==(const SymbolicIndexStamp &rhs) const  {
        return this->index_before_trace == rhs.index_before_trace &&
               this->index_after_trace == rhs.index_after_trace &&
               this->traced_by == rhs.traced_by;
    }

    std::ostream& operator << (std::ostream &os, SymbolicIndexStamp const &its) {
        os << std::string("SymbolicIndexStamp: [Before trace: ")
           << its.index_before_trace
           << std::string(", after trace: ")
           << its.index_after_trace
           << std::string(", traced by: ")
           << its.traced_by
           << std::string("]");

        return os;
    }
}
}
}