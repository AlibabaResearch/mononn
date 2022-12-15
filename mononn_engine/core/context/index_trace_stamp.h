#pragma once
#include <map>
#include <vector>
#include <string>

namespace mononn_engine {
namespace core {
namespace context {
    struct ConcreteIndexStamp {
        std::string index_before_trace;
        std::string index_after_trace;
        std::string traced_by;
        std::string pred_before_trace = "true";
        std::string pred_after_trace = "true";
        std::string value_on_false_pred = "0";

        bool operator == (ConcreteIndexStamp const &rhs) const;

        ConcreteIndexStamp instantiate(const std::map<std::string, std::string> &params) const;

        friend std::ostream& operator << (std::ostream &os, ConcreteIndexStamp const &its);
    };

    struct SymbolicIndexStamp {
        std::string index_before_trace;
        std::string index_after_trace;
        std::string traced_by;
        std::string pred_before_trace = "true";
        std::string pred_after_trace = "true";
        std::string value_on_false_pred = "0";

        ConcreteIndexStamp instantiate(const std::map<std::string, std::string> &params) const;

        bool operator == (SymbolicIndexStamp const &rhs) const;

        friend std::ostream& operator << (std::ostream &os, SymbolicIndexStamp const &its);
    };
}
}
}

