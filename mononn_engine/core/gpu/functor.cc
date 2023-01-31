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

#include "mononn_engine/core/gpu/functor.h"

#include <ctype.h>

#include "mononn_engine/config/config.h"
#include "mononn_engine/helpers/string_helpers.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace core {
namespace gpu {
using OpType = mononn_engine::core::op::OpType;
using Dtype = mononn_engine::core::tensor::Dtype;
using Config = mononn_engine::config::Config;

std::map<std::string, Functor>* Functor::registry() {
  static std::map<std::string, Functor> functor_registry =
      std::map<std::string, Functor>();

  return &functor_registry;
}

std::vector<Functor::Dtype> Functor::supported_types = {
    Functor::Dtype::from_string("bool"),
    Functor::Dtype::from_string("int32"),
    Functor::Dtype::from_string("int64"),
    Functor::Dtype::from_string("float16"),
    Functor::Dtype::from_string("float32"),
};

std::string Functor::get_functor_name_for_op_type(
    OpType op_type, absl::optional<MathOp> math_op) {
  if (op_type == OpType::abs) {
    return "absolute";
  } else if (op_type == OpType::add) {
    return "plus";
  } else if (op_type == OpType::compare) {
    if (math_op.value() == MathOp::equal_to) return "equal";
    if (math_op.value() == MathOp::not_equal_to) return "not_equal";
    if (math_op.value() == MathOp::greater_equal_than) return "greater_equal";
    if (math_op.value() == MathOp::greater_than) return "greater";
    if (math_op.value() == MathOp::less_equal_than) return "less_equal";
    if (math_op.value() == MathOp::less_than) return "less";

    LOG(FATAL) << "Unsupported compare op " << math_op.value().to_string();
  } else if (op_type == OpType::convert) {
    return "convert";
  } else if (op_type == OpType::divide) {
    return "divides";
  } else if (op_type == OpType::exp) {
    return "exponential";
  } else if (op_type == OpType::maximum) {
    return "maximum";
  } else if (op_type == OpType::minimum) {
    return "minimum";
  } else if (op_type == OpType::multiply) {
    return "multiplies";
  } else if (op_type == OpType::rsqrt) {
    return "rsqrt";
  } else if (op_type == OpType::subtract) {
    return "minus";
  } else if (op_type == OpType::select) {
    return "select";
  } else if (op_type == OpType::tanh) {
    return "tanh";
  } else if (op_type == OpType::clamp) {
    return "clamp";
  } else if (op_type == OpType::log) {
    return "natural_log";
  } else if (op_type == OpType::sign) {
    return "sign";
  } else if (op_type == OpType::negate) {
    return "negate";
  }

  LOG(FATAL) << "Do not have functor for op type " << op_type.to_string();
}

class FunctorRegister {
 public:
  FunctorRegister(std::string name) {
    std::map<std::string, Functor>* functor_registry = Functor::registry();

    using Dtype = mononn_engine::core::tensor::Dtype;

    for (Functor::Dtype dtype : Functor::supported_types) {
      // if (dtype == Dtype::from_string("int64") && name != "convert") {
      //     continue;
      // }

      // bool operations.
      if (dtype == Dtype::from_string("bool") && name != "convert" &&
          name != "equal" && name != "not_equal" && name != "sign") {
        continue;
      }

      if ((name == "exponential" || name == "rsqrt" || name == "tanh" ||
           name == "natural_log") &&
          (dtype != Dtype::from_string("float32") &&
           dtype != Dtype::from_string("float16"))) {
        continue;
      }

      if ((name == "tanh" || name == "natural_log") &&
          dtype != Dtype::from_string("float32")) {
        continue;
      }

      Functor functor(name, dtype);

      functor_registry->insert(std::make_pair(functor.get_name(), functor));

      for (int vec_len = 2; vec_len <= 8; vec_len <<= 1) {
        Functor functor_vec(name, dtype.vectorize(vec_len));
        functor_registry->insert(
            std::make_pair(functor_vec.get_name(), functor_vec));
      }

      //                for (int ilp_factor :
      //                Config::get()->candidate_ilp_factor) {
      //                    Functor functor_ilp(name,
      //                    dtype.instruction_parallelize(ilp_factor));
      //                    functor_registry->insert(std::make_pair(functor_ilp.get_name(),
      //                    functor_ilp));
      //                }
    }
  }

 private:
};

#define REGISTER_FUNCTOR(name) FunctorRegister register_functor_##name(#name)

REGISTER_FUNCTOR(absolute);
REGISTER_FUNCTOR(convert);
REGISTER_FUNCTOR(exponential);
REGISTER_FUNCTOR(equal);
REGISTER_FUNCTOR(plus);
REGISTER_FUNCTOR(minus);
REGISTER_FUNCTOR(multiplies);
REGISTER_FUNCTOR(square);
REGISTER_FUNCTOR(select);
REGISTER_FUNCTOR(divides);
REGISTER_FUNCTOR(negate);
REGISTER_FUNCTOR(not_equal);
REGISTER_FUNCTOR(greater_equal);
REGISTER_FUNCTOR(greater);
REGISTER_FUNCTOR(less_equal);
REGISTER_FUNCTOR(less);
REGISTER_FUNCTOR(multiply_add);
REGISTER_FUNCTOR(maximum);
REGISTER_FUNCTOR(minimum);
REGISTER_FUNCTOR(rsqrt);
REGISTER_FUNCTOR(tanh);
REGISTER_FUNCTOR(clamp);
REGISTER_FUNCTOR(natural_log);
REGISTER_FUNCTOR(sign);

std::string Functor::get_definition() const {
  return mononn_engine::helpers::string_format("__device__ %s %s;",
                                               this->get_functor_type().c_str(),
                                               this->get_name().c_str());
}

std::string Functor::get_name() const {
  std::string type_name = this->dtype.to_string();
  for (char& ch : type_name) {
    if (!isalnum(ch)) ch = '_';
  }

  return "functor_" + this->name + "_" + type_name;
}

std::string Functor::get_raw_name() const { return this->name; }

std::string Functor::get_functor_type() const {
  return mononn_engine::helpers::string_format(
      "cutlass::%s<%s>", this->name.c_str(), this->dtype.to_string().c_str());
}

std::string Functor::get_all_functors_definition() {
  std::map<std::string, Functor>* registry = Functor::registry();

  std::stringstream ss;
  for (auto const& [functor_name, functor] : *registry) {
    ss << functor.get_definition();
    ss << "\n";
  }

  return ss.str();
}

Dtype Functor::get_dtype() const { return this->dtype; }
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine