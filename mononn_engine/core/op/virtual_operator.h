#pragma once

// #include <algorithm>
// #include <vector>
// #include <unordered_map>
// #include <memory>

// #include "tensorflow/compiler/xla/service/hlo_instruction.h"
// #include "mononn_engine/core/op_annotation/locality_tier.h"
// #include "mononn_engine/core/op_impl/op_impl_base.h"

namespace mononn_engine {
namespace core {
namespace op {
    // class VirtualOperator {
    // public:
    //     using Tier = mononn_engine::core::op_annotation::LocalityTier::Tier;
    //     using LocalityTier = mononn_engine::core::op_annotation::LocalityTier;
    //     using OpImplBase = mononn_engine::core::op_impl::OpImplBase;
    //     using OpType = mononn_engine::core::op::OpType;
    //     using TUID = std::string;

    //     VirtualOperator() {};

    //     VirtualOperator(xla::HloInstruction *instruction);
    //     TUID get_uid() const;
    // private:
    //     std::vector<Tier> annotation;
    //     std::unordered_map<Tier, std::shared_ptr<OpImplBase>> op_impl;
    //     xla::HloInstruction *instruction;

    //     std::string name;
    //     std::string type;

    //     TUID uid;

    //     // static TUID uid_cnt = 0;
    // };
}
}
}