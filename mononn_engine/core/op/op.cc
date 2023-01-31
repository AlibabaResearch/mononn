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

#include "mononn_engine/core/op/op.h"

#include "mononn_engine/core/gpu/multi_buffer.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/tensor/dtype.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace core {
namespace op {
using OpImpl = mononn_engine::core::op_impl::OpImplBase;
using Tensor = mononn_engine::core::tensor::Tensor;
using TensorSpec = mononn_engine::core::tensor::TensorSpec;
using ClusterType = mononn_engine::core::op_annotation::ClusterType;
using OpType = mononn_engine::core::op::OpType;
using Dtype = mononn_engine::core::tensor::Dtype;
using MultiBuffer = mononn_engine::core::gpu::MultiBuffer;
using SymbolicIndexStamp = mononn_engine::core::context::SymbolicIndexStamp;

const std::string& Op::get_name() const { return this->name; }

std::string Op::get_hlo_instruction_name() const {
  if (this->hlo_instruction_name == "undefined") {
    LOG(WARNING) << this->get_name() << "'s instructionn name not set";
  }

  return this->hlo_instruction_name;
}

void Op::set_hlo_instruction_name(const std::string& _hlo_instruction_name) {
  if (_hlo_instruction_name.empty()) {
    LOG(FATAL) << "inst name cannot be empty in node " << this->get_name();
  }

  this->hlo_instruction_name = _hlo_instruction_name;
}

void Op::set_symbolic_index(
    const std::vector<SymbolicIndexStamp>& _symbolic_index_list) {
  this->symbolic_index_list = _symbolic_index_list;
}

const std::vector<SymbolicIndexStamp>& Op::get_symbolic_index() const {
  return this->symbolic_index_list;
}

// void Op::propagate_index_to_implementation() {
//     this->impl->set_traced_index(this->symbolic_index_list);
// }

std::shared_ptr<OpImpl> Op::get_implementation() const { return this->impl; }

void Op::set_implementation(std::shared_ptr<OpImpl> _impl) {
  this->impl = _impl;
}

std::shared_ptr<Op> Op::get_operand(int index) const {
  return this->operands[index];
}

std::vector<std::shared_ptr<Op>> Op::get_operands() const {
  return this->operands;
}

int Op::get_operand_count() const { return (int)this->operands.size(); }

void Op::replace_operand(std::string old_operand_name,
                         std::shared_ptr<Op> new_operand) {
  for (auto& operand : this->operands) {
    if (operand->get_name() == old_operand_name) {
      if (operand->get_output_specs() != new_operand->get_output_specs()) {
        LOG(FATAL) << "Tensor spec between operand " << old_operand_name
                   << " and " << new_operand->get_name() << " for node "
                   << this->get_name();
      }

      operand = new_operand;
      return;
    }
  }

  LOG(FATAL) << "Operand " << old_operand_name << " not found in node "
             << this->get_name();
}

void Op::set_output_specs(std::vector<TensorSpec> _output_specs) {
  this->output_specs = _output_specs;
}

TensorSpec Op::get_output_spec(int index) const {
  return this->output_specs[index];
}

std::vector<TensorSpec> Op::get_output_specs() const {
  return this->output_specs;
}

int Op::get_output_specs_count() const { return this->output_specs.size(); }

void Op::set_cluster_type(ClusterType _cluster_type) {
  this->cluster_type = _cluster_type;
}

ClusterType Op::get_cluster_type() const { return this->cluster_type; }

void Op::set_cluster_id(int _cluster_id) { this->cluster_id = _cluster_id; }

int Op::get_cluster_id() const { return this->cluster_id; }

std::string Op::get_cluster_name() const {
  return this->cluster_type.to_string() + std::string("_") +
         std::to_string(this->cluster_id);
}

bool Op::is_elewise() const {
  if (this->get_type() != OpType::reduce &&
      this->get_type() != OpType::custom_call &&
      this->get_type() != OpType::convolution)
    return true;
  else
    return false;
}

bool Op::is_gemm() const { return false; }

bool Op::is_conv() const { return false; }

bool Op::is_elewise_unary() const { return false; }

bool Op::is_elewise_binary() const { return false; }

bool Op::is_tuple() const { return false; }

bool Op::is_cluster() const {
  return this->is_cluster_elewise() || this->is_cluster_reduce();
}

bool Op::is_cluster_elewise() const { return false; }

bool Op::is_cluster_reduce() const { return false; }

void Op::set_attribute(const std::string& key, const std::string& value) {
  //        if (value == "") LOG(FATAL) << "Empty attribute of key " << key;
  this->attributes[key] = value;
}

bool Op::has_attribute(const std::string& key) const {
  return this->attributes.find(key) != this->attributes.end();
}

std::string Op::get_attribute(const std::string& key) const {
  if (!this->has_attribute(key))
    LOG(FATAL) << "Node " << this->get_name() << " do not have attribute "
               << key;

  return this->attributes.at(key);
}

void Op::del_attribute(const std::string& key) { this->attributes.erase(key); }

std::unordered_map<std::string, std::string> Op::export_attributes() const {
  return this->attributes;
}

void Op::import_attributes(
    std::unordered_map<std::string, std::string> _attributes) {
  this->attributes = _attributes;
}

bool Op::need_pre_inner_loop_generation() const {
  if (this->impl->need_pre_inner_loop_generation()) return true;

  return false;
}

bool Op::need_post_inner_loop_generation() const {
  if (this->impl->need_post_inner_loop_generation()) return true;

  return false;
}

std::string Op::generate_pre_inner_loop() const {
  return this->impl->generate_pre_inner_loop();
}

std::string Op::generate_post_inner_loop() const {
  return this->impl->generate_post_inner_loop();
}

void Op::set_read_from_global_memory() { this->read_from_global_memory = true; }

bool Op::is_read_from_global_memory() const {
  return this->read_from_global_memory;
}

void Op::set_write_to_global_memory() { this->write_to_global_memory = true; }

bool Op::is_write_to_global_memory() const {
  return this->write_to_global_memory;
}

bool Op::need_generate_with_index() const {
  if (this->get_type() == OpType::gather || this->get_type() == OpType::iota ||
      this->get_type() == OpType::concatenate)
    return true;

  if (this->get_type() == OpType::custom_call ||
      this->get_type() == OpType::get_tuple_element)
    return false;

  // Write to global memory should not generate with index
  // For example: nodes write to global memory in clusters should not generate
  // with index.
  if (this->is_read_from_global_memory()) return true;

  return false;
}

int64_t Op::get_output_buffer_size_in_bytes() const {
  if (this->output_specs.size() == 1) {
    return this->get_output_spec(0).size_in_bytes();
  } else {
    MultiBuffer multi_buffer("fake_name");
    for (auto const& spec : this->output_specs) {
      multi_buffer.add_buffer(spec);
    }

    return multi_buffer.get_total_size_in_bytes();
  }
}

Tensor Op::get_output_tensor(int index) const {
  return Tensor(this->get_name(), this->get_output_spec(index));
}

void Op::set_hlo_text(std::string text) { this->hlo_dumped_text = text; }

std::string Op::get_hlo_text() const { return this->hlo_dumped_text; }

bool Op::vectorized() const {
  return this->get_output_spec(0).get_dtype().is_vectorized();
}

void Op::vectorize(int vec_len) {
  EXPECT_TRUE(this->get_output_specs().size() == 1,
              this->get_name() + " have multiple outputs.");
  EXPECT_TRUE(!this->vectorized(), this->get_name() + " already vectorized.");
  TensorSpec vectorized_spec = this->get_output_spec(0).vectorize(vec_len);

  this->set_output_specs({vectorized_spec});
}

void Op::assert_single_output() const {
  EXPECT_TRUE(this->get_output_specs().size() == 1,
              "Node " + this->get_name() + " should have exactly one output.");
}

// bool Op::can_vectorize_input_as(int vec_len, int input_id) const {
//     this->assert_single_output();
//     EXPECT_TRUE(input_id < this->get_operand_count(), "Input id out of
//     range");

//     if (this->get_operand(input_id)->get_output_spec(0).get_shape(-1) %
//     vec_len != 0) return false;

//     return this->can_vectorize_input_as_impl(vec_len, input_id);
// }

// bool Op::can_vectorize_output_as(int vec_len) const {
//     this->assert_single_output();
//     if (this->vectorized()) return false;

//     if (this->get_output_spec(0).get_shape(-1) % vec_len != 0) return false;

//     return this->can_vectorize_output_as_impl(vec_len);
// }

// int Op::vectorization_propagate_to_input(int vec_len, int input_id) const {
//     this->assert_single_output();
//     EXPECT_TRUE(this->can_vectorize_output_as(vec_len),
//                 "Cannot vectorize output of node " + this->get_name() + " to
//                 vector len " + std::to_string(vec_len));
//     EXPECT_TRUE(input_id < this->get_operand_count(), "Input id out of
//     range"); return this->vectorization_propagate_to_input_impl(vec_len,
//     input_id);
// }

// int Op::vectorization_propagate_to_output(int vec_len, int input_id) const {
//     this->assert_single_output();
//     EXPECT_TRUE(this->can_vectorize_input_as(vec_len, input_id),
//                 "Cannot vectorize input of node " + this->get_name() + " to
//                 vector len " + std::to_string(vec_len));
//     return this->vectorization_propagate_to_output_impl(vec_len);
// }

void Op::set_value(std::string _value) { LOG(FATAL) << "Un implemented"; }

std::string Op::get_value() const { LOG(FATAL) << "Un implemented"; }

void Op::add_auxiliary_impl(std::string key, std::shared_ptr<OpImpl> _impl) {
  this->impl->add_auxiliary_impl(key, _impl);
}

void Op::instantiate_concrete_index(
    const std::map<std::string, std::string>& params,
    const std::string& loop_stride) {
  this->impl->instantiate_concrete_index(this->symbolic_index_list, params,
                                         loop_stride);
}

void Op::instantiate_ilp_concrete_index(
    const std::map<std::string, std::string>& params,
    const std::string& loop_stride, const std::string& ilp_stride) {
  this->impl->instantiate_ilp_concrete_index(this->symbolic_index_list, params,
                                             loop_stride, ilp_stride);
}

void Op::set_instruction_parallel_factor(int _ilp_factor) {
  this->ilp_factor = _ilp_factor;

  this->impl->set_instruction_parallel_factor(_ilp_factor);
}

// void Op::deep_copy_base_op(Op *new_op) const {
//     new_op->name = this->name;
//     new_op->hlo_instruction_name = this->hlo_instruction_name;
//     new_op->impl =
//     std::make_shared<OpImpl>(this->impl->deep_copy().release());
//     new_op->output_specs = this->output_specs;
//     new_op->cluster_type = this->cluster_type;
//     new_op->hlo_dumped_text = this->hlo_dumped_text;
//     new_op->cluster_id = this->cluster_id;
//     new_op->read_from_global_memory = this->read_from_global_memory;
//     new_op->write_to_global_memory = this->write_to_global_memory;
// }

// void Op::propagate_ilp_index_to_implementation() {
//     if (!this->impl->is_instruction_parallelized()) {
//         LOG(FATAL) << "Node " << this->get_name() << "'s implementation has
//         not been instruction parallelized yet";
//     }

//     if (this->get_instruction_parallel_factor() !=
//     this->impl->get_instruction_parallel_factor()) {
//         LOG(FATAL) << "Node " << this->get_name() << " instruction parallel
//         factor does not match";
//     }

//     for (int idx = 0; idx < this->ilp_factor; ++idx) {
//         this->impl->set_ilp_traced_index(idx,
//         this->get_ilp_traced_index(idx));
//     }
// }

// bool Op::can_vectorize_input_as_impl(int vec_len, int input_id) const {
//     return true;
// }

// bool Op::can_vectorize_output_as_impl(int vec_len) const {
//     return true;
// }

// int Op::vectorization_propagate_to_input_impl(int vec_len, int input_id)
// const {
//     return vec_len;
// }

// int Op::vectorization_propagate_to_output_impl(int vec_len) const {
//     return vec_len;
// }
}  // namespace op
}  // namespace core
}  // namespace mononn_engine