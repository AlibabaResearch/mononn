// Copyright 2023 The MonoNN Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
    Tier() : tier(-1){};
    Tier(int _tier) : tier(_tier) {}

    std::string to_string() const {
      return mononn_engine::helpers::string_format("Tier%d", this->tier);
    }

    bool operator<(const Tier& rhs) const { return this->tier < rhs.tier; }

    bool operator==(const Tier& rhs) const { return this->tier == rhs.tier; }

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
}  // namespace op_annotation
}  // namespace core
}  // namespace mononn_engine