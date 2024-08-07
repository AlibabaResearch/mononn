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

#include "mononn_engine/core/op/virtual_operator.h"

namespace mononn_engine {
namespace core {
namespace op {
// VirtualOperator::VirtualOperator(xla::HloInstruction *instruction) {
//     std::string inst_name = instruction->name();
//     bool is_t0_op = std::any_of(LocalityTier::OpT0.begin(),
//     LocalityTier::OpT0.end(), [&](const std::string &op)-> bool {
//         return inst_name.find(op) == 0;
//     });

//     bool is_t1_op = std::any_of(LocalityTier::OpT1.begin(),
//     LocalityTier::OpT1.end(), [&](const std::string &op)-> bool {
//         return inst_name.find(op) == 0;
//     });

//     bool is_t2_op = std::any_of(LocalityTier::OpT2.begin(),
//     LocalityTier::OpT2.end(), [&](const std::string &op)-> bool {
//         return inst_name.find(op) == 0;
//     });

//     bool is_t3_op = std::any_of(LocalityTier::OpT3.begin(),
//     LocalityTier::OpT3.end(), [&](const std::string &op)-> bool {
//         return inst_name.find(op) == 0;
//     });

//     if (is_t0_op) annotation.push_back(LocalityTier::kT0);
//     if (is_t1_op) annotation.push_back(LocalityTier::kT1);
//     if (is_t2_op) annotation.push_back(LocalityTier::kT2);
//     if (is_t3_op) annotation.push_back(LocalityTier::kT3);

//     this->name = inst_name;
//     if (inst_name.find(OpType::constant) == 0) {
//         this->type = inst_name.substr(0, inst_name.find("-"));
//     } else {
//         this->type = inst_name.substr(0, inst_name.find("."));
//     }

//     this->uid = inst_name;
// }

// VirtualOperator::TUID VirtualOperator::get_uid() const {
//     return this->uid;
// }
}  // namespace op
}  // namespace core
}  // namespace mononn_engine
