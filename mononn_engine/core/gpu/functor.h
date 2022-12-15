#pragma once 
#include <string>
#include <vector>
#include <map>
#include <unordered_map>

#include "mononn_engine/core/tensor/dtype.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/tensor/math_op.h"
#include "absl/types/optional.h"

namespace mononn_engine {
namespace core {
namespace gpu {
    class Functor {
    public:
        using Dtype = mononn_engine::core::tensor::Dtype;
        using OpType = mononn_engine::core::op::OpType;
        using MathOp = mononn_engine::core::tensor::MathOp;


        Functor() {}
        Functor(std::string _name, Dtype _dtype) : name(_name), dtype(_dtype) {}
        Functor(const char *_name, Dtype _dtype) : name(std::string(_name)), dtype(_dtype) {}

        static std::map<std::string, Functor> *registry();
        static std::vector<Dtype> supported_types;
        static std::string get_functor_name_for_op_type(OpType op_type, absl::optional<MathOp> math_op = absl::nullopt);
        static std::string get_all_functors_definition();

        std::string get_definition() const;
        std::string get_name() const;
        std::string get_raw_name() const;
        std::string get_functor_type() const;
        Dtype get_dtype() const;
    private:
        std::string name;
        Dtype dtype;
    };
}
}
}