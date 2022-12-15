#include "mononn_engine/core/tensor/math_op.h"
#include "mononn_engine/helpers/macros.h"

namespace mononn_engine {
namespace core {
namespace tensor {
    std::unordered_map<std::string, MathOp>* MathOp::registry() {
        static std::unordered_map<std::string, MathOp> op_registry = 
            std::unordered_map<std::string, MathOp>();

        return &op_registry;
    }

    struct MathOpRegister {
        MathOpRegister() = delete;
        MathOpRegister(MathOp op) {
            MathOp::registry()->insert(std::make_pair(op.to_string(), op));
        }
    };

    #define REGISTER_MATH_OP_UNIQUE(ctr, op) \
        MathOpRegister register_math_op_unique_##ctr(op)

    #define REGISTER_MATH_OP_UNIQUE_HELPER(ctr, op) \
        REGISTER_MATH_OP_UNIQUE(ctr, op)

    #define REGISTER_MATH_OP(op) \ 
        REGISTER_MATH_OP_UNIQUE_HELPER(__COUNTER__, op)
    
    const MathOp MathOp::assign = "=";
    const MathOp MathOp::plus_assign = "+=";
    const MathOp MathOp::equal_to = "==";
    const MathOp MathOp::not_equal_to = "!=";
    const MathOp MathOp::greater_equal_than = ">=";
    const MathOp MathOp::greater_than = ">";
    const MathOp MathOp::less_equal_than = "<=";
    const MathOp MathOp::less_than = "<";

    REGISTER_MATH_OP(MathOp::assign);
    REGISTER_MATH_OP(MathOp::plus_assign);
    REGISTER_MATH_OP(MathOp::equal_to);
    REGISTER_MATH_OP(MathOp::not_equal_to);
    REGISTER_MATH_OP(MathOp::greater_equal_than);
    REGISTER_MATH_OP(MathOp::greater_than);
    REGISTER_MATH_OP(MathOp::less_equal_than);
    REGISTER_MATH_OP(MathOp::less_than);


    std::string MathOp::to_string() const {
        return this->op;
    }

    MathOp MathOp::from_string(std::string str) {
        EXPECT_TRUE(MathOp::registry()->find(str) != MathOp::registry()->end(), "Unexpected op: " + str);
        return MathOp::registry()->at(str);
    }

    bool MathOp::operator == (MathOp const &rhs) const {
        return this->op == rhs.op;
    }
}
}
}