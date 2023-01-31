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

#include "mononn_engine/core/tensor/math_op.h"
#include "mononn_engine/core/tensor/scalar.h"
#include "mononn_engine/core/tensor/tensor_shape.h"

namespace mononn_engine {
namespace core {
namespace schedule {
class Loop {
 public:
  using Scalar = mononn_engine::core::tensor::Scalar;
  using TensorShape = mononn_engine::core::tensor::TensorShape;
  using MathOp = mononn_engine::core::tensor::MathOp;

  class Condition {
   public:
    Condition() {}
    Condition(std::string _left_statement, std::string _right_statement,
              MathOp _cmp)
        : left_statement(_left_statement),
          right_statement(_right_statement),
          cmp(_cmp) {}

    std::string get_left_statement() const { return this->left_statement; }

    std::string get_right_statement() const { return this->right_statement; }

    MathOp get_cmp() const { return this->cmp; }

    std::string to_string() const {
      return "(" + this->get_left_statement() + " " +
             this->get_cmp().to_string() + " " + this->get_right_statement() +
             ")";
    }

    static Condition less_than(std::string left_statement,
                               std::string right_statement);

   private:
    std::string left_statement;
    std::string right_statement;
    MathOp cmp;
  };

  Loop(TensorShape _loop_shape, Scalar _key, std::string _init,
       std::string _stride)
      : loop_shape(_loop_shape), key(_key), init(_init), stride(_stride) {
    this->condition = Condition(
        this->key.get_name(), std::to_string(this->loop_shape.element_count()),
        MathOp::less_than);
  }

  TensorShape get_loop_shape() const;
  Scalar get_loop_key() const;
  std::string get_loop_steps() const;
  std::string get_loop_step_id() const;
  std::string get_loop_stride() const;
  std::string get_loop_ilp_stride() const;
  Condition get_loop_condition() const;
  std::string get_loop_init() const;

  std::string begin_loop() const;
  std::string end_loop() const;

  Loop vectorize(int elements_per_access) const;
  Loop instruction_level_parallelism(int _ilp_factor);
  void set_unroll(bool unroll, int factor);

  bool is_instruction_parallelized() const;
  void set_ilp_factor(int _ilp_factor);

 private:
  TensorShape loop_shape;
  Scalar key;
  std::string init;
  Condition condition;
  std::string stride;
  bool unroll = false;
  int unroll_factor = -1;
  int ilp_factor = 1;
};
}  // namespace schedule
}  // namespace core
}  // namespace mononn_engine