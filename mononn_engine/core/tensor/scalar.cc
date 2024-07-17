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

#include "mononn_engine/core/tensor/scalar.h"

#include "mononn_engine/helpers/string_helpers.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace core {
namespace tensor {

Scalar::Scalar(const std::string& _name, const std::vector<Dtype>& _dtype,
               const std::vector<std::string>& _value)
    : name(_name), dtype(_dtype), value(_value) {
  if (this->dtype.size() != this->value.size()) {
    LOG(FATAL) << "Tuple element number does not match: " << this->dtype.size()
               << " vs " << this->value.size();
  }
}

std::string Scalar::get_definition() const {
  return mononn_engine::helpers::string_format(
      "%s %s;", this->get_type_string().c_str(), this->get_name().c_str());
}

std::string Scalar::get_definition_with_value() const {
  if (this->get_value().length() == 0) {
    LOG(FATAL) << "Scalar do not have value";
  }

  return mononn_engine::helpers::string_format(
      "%s %s = %s;", this->get_type_string().c_str(), this->get_name().c_str(),
      this->get_value().c_str());
}

std::string Scalar::get_name() const { return this->name; }

std::string Scalar::get_value() const {
  std::vector<std::string> value_list;
  for (int tuple_idx = 0; tuple_idx < this->dtype.size(); ++tuple_idx) {
    value_list.push_back(this->value[tuple_idx]);
  }

  std::string ret_value = mononn_engine::helpers::join(", ", value_list);

  if (value_list.size() > 1) {
    ret_value = "cuda::std::make_tuple(" + ret_value + ")";
  }

  return ret_value;
}

const Dtype& Scalar::get_dtype() const {
  if (this->is_tuple()) {
    LOG(FATAL) << "Is tuple";
  }

  return this->dtype.at(0);
}

std::string Scalar::get_type_string() const {
  if (this->is_tuple()) {
    std::vector<std::string> type_list;
    for (int tuple_idx = 0; tuple_idx < this->dtype.size(); ++tuple_idx) {
      type_list.push_back(this->dtype[tuple_idx].to_string());
    }

    std::string type_string = "cuda::std::tuple<" +
                              mononn_engine::helpers::join(", ", type_list) +
                              ">";

    return type_string;
  } else {
    return this->dtype.at(0).to_string();
  }
}

bool Scalar::is_tuple() const { return this->dtype.size() > 1; }

const std::vector<Dtype>& Scalar::get_types_in_list() const {
  return this->dtype;
}

const std::vector<std::string>& Scalar::get_values_in_list() const {
  return this->value;
}

// Scalar Scalar::vectorize(int _element_per_access) const {
//     return Scalar(
//             this->name,
//             this->dtype.vectorize(_element_per_access),
//             this->value
//     );

// }
}  // namespace tensor
}  // namespace core
}  // namespace mononn_engine