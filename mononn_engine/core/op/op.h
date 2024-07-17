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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "mononn_engine/core/common/ilp_node_interface.h"
#include "mononn_engine/core/common/pointer_convert.h"
#include "mononn_engine/core/context/cuda_context.h"
#include "mononn_engine/core/context/index_trace_stamp.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/op_annotation/cluster_type.h"
#include "mononn_engine/core/op_annotation/locality_tier.h"
#include "mononn_engine/core/op_impl/op_impl_base.h"
#include "mononn_engine/core/tensor/tensor.h"
#include "mononn_engine/core/tensor/tensor_spec.h"

namespace mononn_engine {
namespace core {
namespace op {
using ILPNodeInterface = mononn_engine::core::common::ILPNodeInterface;
using PointerConvert = mononn_engine::core::common::PointerConvert;

class Op : public ILPNodeInterface, public PointerConvert {
 public:
  using OpImpl = mononn_engine::core::op_impl::OpImplBase;
  using Tensor = mononn_engine::core::tensor::Tensor;
  using TensorSpec = mononn_engine::core::tensor::TensorSpec;
  using CUDAContext = mononn_engine::core::context::CUDAContext;
  using ClusterType = mononn_engine::core::op_annotation::ClusterType;
  using Tier = mononn_engine::core::op_annotation::LocalityTier::Tier;
  using OpType = mononn_engine::core::op::OpType;
  using SymbolicIndexStamp = mononn_engine::core::context::SymbolicIndexStamp;

  Op(std::string _name, std::vector<std::shared_ptr<Op>> _operands,
     std::vector<TensorSpec> _output_specs)
      : name(_name), operands(_operands), output_specs(_output_specs) {}

  const std::string& get_name() const;
  std::string get_hlo_instruction_name() const;
  void set_hlo_instruction_name(const std::string& _hlo_instruction_name);

  void set_symbolic_index(
      const std::vector<SymbolicIndexStamp>& _symbolic_index_list);
  const std::vector<SymbolicIndexStamp>& get_symbolic_index() const;
  // void propagate_index_to_implementation();

  virtual OpType get_type() const = 0;
  virtual std::vector<std::shared_ptr<OpImpl>>
  generate_candidate_implementation(std::shared_ptr<CUDAContext> context,
                                    Tier tier) const = 0;

  virtual std::shared_ptr<OpImpl> get_implementation() const;
  virtual void set_implementation(std::shared_ptr<OpImpl> _impl);

  std::shared_ptr<Op> get_operand(int index) const;
  std::vector<std::shared_ptr<Op>> get_operands() const;
  int get_operand_count() const;
  virtual void replace_operand(std::string old_operand_name,
                               std::shared_ptr<Op> new_operand);

  void set_output_specs(std::vector<TensorSpec> _output_specs);
  TensorSpec get_output_spec(int index) const;
  std::vector<TensorSpec> get_output_specs() const;
  int get_output_specs_count() const;

  Tensor get_output_tensor(int index) const;

  virtual void set_cluster_type(ClusterType _cluster_type);
  virtual ClusterType get_cluster_type() const;

  void set_cluster_id(int _cluster_id);
  int get_cluster_id() const;

  std::string get_cluster_name() const;

  virtual bool is_elewise() const;
  virtual bool is_gemm() const;
  virtual bool is_conv() const;
  virtual bool is_elewise_unary() const;
  virtual bool is_elewise_binary() const;
  virtual bool is_tuple() const;
  virtual bool is_cluster() const;
  virtual bool is_cluster_elewise() const;
  virtual bool is_cluster_reduce() const;

  void set_attribute(const std::string& key, const std::string& value);
  bool has_attribute(const std::string& key) const;
  std::string get_attribute(const std::string& key) const;
  void del_attribute(const std::string& key);

  std::unordered_map<std::string, std::string> export_attributes() const;
  void import_attributes(
      std::unordered_map<std::string, std::string> _attributes);

  virtual bool need_pre_inner_loop_generation() const;
  virtual bool need_post_inner_loop_generation() const;
  virtual std::string generate_pre_inner_loop() const;
  virtual std::string generate_post_inner_loop() const;

  void set_read_from_global_memory();
  bool is_read_from_global_memory() const;

  void set_write_to_global_memory();
  bool is_write_to_global_memory() const;

  bool need_generate_with_index() const;

  int64_t get_output_buffer_size_in_bytes() const;

  void set_hlo_text(std::string text);
  std::string get_hlo_text() const;

  bool vectorized() const;
  void vectorize(int vec_len);
  void assert_single_output() const;

  // bool can_vectorize_input_as(int vec_len, int input_id) const;
  // bool can_vectorize_output_as(int vec_len) const;
  // int vectorization_propagate_to_input(int vec_len, int input_id) const;
  // int vectorization_propagate_to_output(int vec_len, int input_id) const;

  virtual void set_value(std::string _value);
  virtual std::string get_value() const;

  void add_auxiliary_impl(std::string key, std::shared_ptr<OpImpl> _impl);

  void instantiate_concrete_index(
      const std::map<std::string, std::string>& params,
      const std::string& loop_stride);
  void instantiate_ilp_concrete_index(
      const std::map<std::string, std::string>& params,
      const std::string& loop_stride, const std::string& ilp_stride);

  virtual void set_instruction_parallel_factor(int _ilp_factor) override;
  // virtual void propagate_ilp_index_to_implementation();

  // void deep_copy_base_op(Op *new_op) const;

 protected:
  // virtual bool can_vectorize_input_as_impl(int vec_len, int input_id) const;
  // virtual bool can_vectorize_output_as_impl(int vec_len) const;
  // virtual int vectorization_propagate_to_input_impl(int vec_len, int
  // input_id) const; virtual int vectorization_propagate_to_output_impl(int
  // vec_len) const;
 private:
  std::string name;
  std::string hlo_instruction_name = "undefined";
  std::shared_ptr<OpImpl> impl;

  std::vector<TensorSpec> output_specs;
  ClusterType cluster_type = ClusterType::None;
  std::string hlo_dumped_text;
  int cluster_id = -1;
  bool read_from_global_memory = false;
  bool write_to_global_memory = false;

 protected:
  std::vector<SymbolicIndexStamp> symbolic_index_list;
  std::unordered_map<std::string, std::string> attributes;
  std::vector<std::shared_ptr<Op>> operands;
};
}  // namespace op
}  // namespace core
}  // namespace mononn_engine