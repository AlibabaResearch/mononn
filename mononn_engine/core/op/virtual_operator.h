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
}  // namespace op
}  // namespace core
}  // namespace mononn_engine