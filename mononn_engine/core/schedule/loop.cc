#include "mononn_engine/core/schedule/loop.h"

#include <sstream>

#include "mononn_engine/helpers/string_helpers.h"

namespace mononn_engine {
namespace core {
namespace schedule {
using Scalar = mononn_engine::core::tensor::Scalar;

Loop::TensorShape Loop::get_loop_shape() const { return this->loop_shape; }

std::string Loop::begin_loop() const {
  std::stringstream ss;

  if (this->unroll) {
    ss << "#pragma unroll";
    if (!this->unroll_factor != -1) ss << "(" << this->unroll_factor << ")\n";
  }

  ss << mononn_engine::helpers::string_format(
      "for (%s %s = %s; %s; %s += %s) {\n",
      this->key.get_dtype().to_string().c_str(), this->key.get_name().c_str(),
      this->init.c_str(), this->condition.to_string().c_str(),
      this->key.get_name().c_str(),
      this->is_instruction_parallelized() ? this->get_loop_ilp_stride().c_str()
                                          : this->stride.c_str());

  return ss.str();
}

std::string Loop::end_loop() const { return "}\n"; }

Loop Loop::vectorize(int elements_per_access) const {
  std::vector<int> new_shape = this->loop_shape.get_shape();
  EXPECT_TRUE(new_shape.back() % elements_per_access == 0,
              "Bad format for loop vectorization");
  new_shape.back() = new_shape.back() / elements_per_access;

  TensorShape new_loop_shape = new_shape;
  Scalar new_key = this->key;
  std::string new_init = this->init;
  std::string new_stride = this->stride;

  return Loop(new_loop_shape, new_key, new_init, new_stride);
}

Loop Loop::instruction_level_parallelism(int _ilp_factor) {
  Loop new_loop(this->get_loop_shape(), this->key, this->init, this->stride);
  new_loop.set_ilp_factor(_ilp_factor);

  return new_loop;
}

void Loop::set_unroll(bool unroll, int factor) {
  this->unroll = unroll;
  this->unroll_factor = factor;
}

bool Loop::is_instruction_parallelized() const { return this->ilp_factor != 1; }

void Loop::set_ilp_factor(int _ilp_factor) { this->ilp_factor = _ilp_factor; }

std::string Loop::get_loop_steps() const {
  return mononn_engine::helpers::string_format(
      "((%s) + (%s) - 1) / (%s)", this->condition.get_right_statement().c_str(),
      this->stride.c_str(), this->stride.c_str());
}

std::string Loop::get_loop_step_id() const {
  return mononn_engine::helpers::string_format(
      "(%s / %s)", this->key.get_name().c_str(), this->stride.c_str());
}

std::string Loop::get_loop_stride() const { return this->stride; }

std::string Loop::get_loop_ilp_stride() const {
  return mononn_engine::helpers::string_format(
      "(%s) * (%s)", this->stride.c_str(),
      std::to_string(this->ilp_factor).c_str());
}

Scalar Loop::get_loop_key() const { return this->key; }

Loop::Condition Loop::get_loop_condition() const { return this->condition; }

std::string Loop::get_loop_init() const { return this->init; }

Loop::Condition Loop::Condition::less_than(std::string left_statement,
                                           std::string right_statement) {
  return Loop::Condition(left_statement, right_statement,
                         Loop::MathOp::less_than);
}
}  // namespace schedule
}  // namespace core
}  // namespace mononn_engine