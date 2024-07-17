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

#include "mononn_engine/core/op_annotation/locality_tier.h"

namespace mononn_engine {
namespace core {
namespace op_annotation {
const LocalityTier::Tier LocalityTier::kT0 = 0;
const LocalityTier::Tier LocalityTier::kT1 = 1;
const LocalityTier::Tier LocalityTier::kT2 = 2;
const LocalityTier::Tier LocalityTier::kT3 = 3;
using OpType = mononn_engine::core::op::OpType;

std::unordered_map<std::string, OpType>* LocalityTier::get_OpT0() {
  static std::unordered_map<std::string, OpType>* registry = nullptr;

  if (registry == nullptr) {
    registry = new std::unordered_map<std::string, OpType>;

    registry->insert(std::make_pair(OpType::abs.get_name(), OpType::abs));
    registry->insert(std::make_pair(OpType::add.get_name(), OpType::add));
    registry->insert(
        std::make_pair(OpType::bitcast.get_name(), OpType::bitcast));
    registry->insert(
        std::make_pair(OpType::broadcast.get_name(), OpType::broadcast));
    registry->insert(std::make_pair(OpType::clamp.get_name(), OpType::clamp));
    registry->insert(
        std::make_pair(OpType::concatenate.get_name(), OpType::concatenate));
    registry->insert(
        std::make_pair(OpType::constant.get_name(), OpType::constant));
    registry->insert(
        std::make_pair(OpType::convert.get_name(), OpType::convert));
    registry->insert(std::make_pair(OpType::copy.get_name(), OpType::copy));
    registry->insert(std::make_pair(OpType::divide.get_name(), OpType::divide));
    registry->insert(std::make_pair(OpType::exp.get_name(), OpType::exp));
    registry->insert(std::make_pair(OpType::gather.get_name(), OpType::gather));
    registry->insert(std::make_pair(OpType::get_tuple_element.get_name(),
                                    OpType::get_tuple_element));
    registry->insert(std::make_pair(OpType::iota.get_name(), OpType::iota));
    registry->insert(
        std::make_pair(OpType::maximum.get_name(), OpType::maximum));
    registry->insert(
        std::make_pair(OpType::minimum.get_name(), OpType::minimum));
    registry->insert(
        std::make_pair(OpType::multiply.get_name(), OpType::multiply));
    registry->insert(std::make_pair(OpType::pad.get_name(), OpType::pad));
    registry->insert(
        std::make_pair(OpType::parameter.get_name(), OpType::parameter));
    registry->insert(std::make_pair(OpType::reduce.get_name(), OpType::reduce));
    registry->insert(
        std::make_pair(OpType::reshape.get_name(), OpType::reshape));
    registry->insert(std::make_pair(OpType::rsqrt.get_name(), OpType::rsqrt));
    registry->insert(std::make_pair(OpType::select.get_name(), OpType::select));
    registry->insert(std::make_pair(OpType::slice.get_name(), OpType::slice));
    registry->insert(
        std::make_pair(OpType::subtract.get_name(), OpType::subtract));
    registry->insert(std::make_pair(OpType::tanh.get_name(), OpType::tanh));
    registry->insert(std::make_pair(OpType::tuple.get_name(), OpType::tuple));
  }

  return registry;
}

std::unordered_map<std::string, OpType>* LocalityTier::get_OpT1() {
  static std::unordered_map<std::string, OpType>* registry = nullptr;

  if (registry == nullptr) {
    registry = new std::unordered_map<std::string, OpType>;
    registry->insert(
        std::make_pair(OpType::custom_call.get_name(), OpType::custom_call));
    registry->insert(std::make_pair(OpType::reduce.get_name(), OpType::reduce));
  }

  return registry;
}

std::unordered_map<std::string, OpType>* LocalityTier::get_OpT2() {
  static std::unordered_map<std::string, OpType>* registry = nullptr;

  if (registry == nullptr) {
    registry = new std::unordered_map<std::string, OpType>;
    registry->insert(
        std::make_pair(OpType::custom_call.get_name(), OpType::custom_call));
    registry->insert(std::make_pair(OpType::reduce.get_name(), OpType::reduce));
  }

  return registry;
}

std::unordered_map<std::string, OpType>* LocalityTier::get_OpT3() {
  static std::unordered_map<std::string, OpType>* registry = nullptr;

  if (registry == nullptr) {
    registry = new std::unordered_map<std::string, OpType>;
    registry->insert(
        std::make_pair(OpType::custom_call.get_name(), OpType::custom_call));
    registry->insert(std::make_pair(OpType::reduce.get_name(), OpType::reduce));
  }

  return registry;
}

// const std::array<std::string, 25> LocalityTier::OpT0 = {
//     {
//         OpType::add,
//         OpType::arg,
//         OpType::bitcast,
//         OpType::broadcast,
//         OpType::clamp,
//         OpType::concatenate,
//         OpType::constant,
//         OpType::convert,
//         OpType::copy,
//         OpType::divide,
//         OpType::exponential,
//         OpType::gather,
//         OpType::get_tuple_element,
//         OpType::iota,
//         OpType::maximum,
//         OpType::multiply,
//         OpType::pad,
//         OpType::reduce,
//         OpType::reshape,
//         OpType::rsqrt,
//         OpType::select,
//         OpType::slice,
//         OpType::subtract,
//         OpType::tanh,
//         OpType::tuple
//     }
// };

// const std::array<std::string, 2> LocalityTier::OpT1 = {
//     {
//         OpType::custom_call,
//         OpType::reduce
//     }
// };

// const std::array<std::string, 2> LocalityTier::OpT2 = {
//     {
//         OpType::custom_call,
//         OpType::reduce
//     }
// };

// const std::array<std::string, 2> LocalityTier::OpT3 = {
//     {
//         OpType::custom_call,
//         OpType::reduce
//     }
// };
}  // namespace op_annotation
}  // namespace core
}  // namespace mononn_engine