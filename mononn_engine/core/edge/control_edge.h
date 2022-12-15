#pragma once
#include <memory>
#include <string>

#include "mononn_engine/core/op/op.h"

namespace mononn_engine {
namespace core {
namespace edge {
    class ControlEdge {
    public:
        using Op = mononn_engine::core::op::Op;

        ControlEdge(std::shared_ptr<Op> _src, std::shared_ptr<Op> _dst) :
            src(_src), dst(_dst) {}

        std::shared_ptr<Op> get_src();
        std::shared_ptr<const Op> get_src() const;
        std::shared_ptr<Op> get_dst();
        std::shared_ptr<const Op> get_dst() const;

        std::string get_src_name() const;
        std::string get_dst_name() const;

        std::string to_string() const;

    private:
        std::shared_ptr<Op> src;
        std::shared_ptr<Op> dst;
    };
}
}
}