#pragma once
#include <string>
#include <vector>
#include <memory>

#include "mononn_engine/core/tensor/tensor_spec.h"
#include "mononn_engine/core/op/op.h"
#include "mononn_engine/core/op/op_type.h"

namespace mononn_engine {
namespace core {
namespace op {
    class Output : public Op {
    public:
        using TensorSpec = mononn_engine::core::tensor::TensorSpec;
        using Tier = mononn_engine::core::op_annotation::LocalityTier::Tier;

        Output(std::string _name, std::vector<std::shared_ptr<Op>> _operands, std::vector<TensorSpec> _output_specs)
                : Op(_name, _operands, _output_specs) {}
        OpType get_type() const override;
        std::vector<std::shared_ptr<OpImpl>> generate_candidate_implementation(std::shared_ptr<CUDAContext> context, Tier tier) const override;

        void set_output_number(int _output_number);
        int get_output_number() const;

    private:

        int output_number;
    };
}
}
}