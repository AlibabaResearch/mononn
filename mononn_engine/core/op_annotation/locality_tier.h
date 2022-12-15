#pragma once
#include <string>
#include <unordered_map>
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/helpers/string_helpers.h"

namespace mononn_engine {
namespace core {
namespace op_annotation {
    struct LocalityTier {
        using OpType = mononn_engine::core::op::OpType;

        struct Tier {
            Tier() : tier(-1) {};
            Tier(int _tier) : tier(_tier) {}

            std::string to_string() const {
                return mononn_engine::helpers::string_format("Tier%d", this->tier);
            }

            bool operator < (const Tier &rhs) const {
                return this->tier < rhs.tier;
            }

            bool operator == (const Tier &rhs) const {
                return this->tier == rhs.tier;
            }

            int tier;
        };


        static const Tier kT0;
        static const Tier kT1;
        static const Tier kT2;
        static const Tier kT3;

        static std::unordered_map<std::string, OpType>* get_OpT0(); 
        static std::unordered_map<std::string, OpType>* get_OpT1(); 
        static std::unordered_map<std::string, OpType>* get_OpT2(); 
        static std::unordered_map<std::string, OpType>* get_OpT3(); 
    };
}
}
}